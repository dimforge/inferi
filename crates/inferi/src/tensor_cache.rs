use dashmap::DashMap;
use khal::backend::{DeviceValue, GpuBackendError};
use khal::BufferUsages;
use std::any::{Any, TypeId};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use vortx::tensor::{AsTensorMut, AsTensorRef, Tensor, TensorMut, TensorRef};
// HACK: this is a last-minute workaround to keep tensors alive so they don’t get freed before
//       the pipeline runs when using the `LlmContext`. Need to revisit this after the conference.

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct TensorKey {
    ty: TypeId,
    shape: [u32; 4],
    rank: u32,
    usage: BufferUsages,
}

impl TensorKey {
    pub fn with_type<T: Any>(shape: &[u32], usage: BufferUsages) -> TensorKey {
        let rank = shape.len();
        let mut padded_shape = [1; 4];
        padded_shape[..rank].copy_from_slice(shape);
        Self {
            ty: TypeId::of::<T>(),
            shape: padded_shape,
            rank: rank as u32,
            usage,
        }
    }
    pub fn new<T: DeviceValue>(tensor: &Tensor<T>, usage: BufferUsages) -> TensorKey {
        TensorKey {
            ty: TypeId::of::<T>(),
            shape: tensor.shape(),
            rank: tensor.rank(),
            usage,
        }
    }
}

pub struct CachedTensor<T: DeviceValue> {
    tensor: Option<Box<Tensor<T>>>, // Use an Option to move-out the tensor on drop.
    cache: TensorCache,
    usage: BufferUsages,
}

impl<T: DeviceValue> CachedTensor<T> {
    pub fn tensor(&self) -> &Tensor<T> {
        self.tensor
            .as_ref()
            .expect("internal error: tensor was already dropped")
    }

    pub fn tensor_mut(&mut self) -> &mut Tensor<T> {
        self.tensor
            .as_mut()
            .expect("internal error: tensor was already dropped")
    }

    // TODO: return the naked tensor once `Box::into_inner` is stabilized.
    pub fn into_inner(mut self) -> Box<Tensor<T>> {
        self.tensor
            .take()
            .expect("internal error: tensor was already dropped")
    }
}

impl<T: DeviceValue> AsRef<Tensor<T>> for CachedTensor<T> {
    fn as_ref(&self) -> &Tensor<T> {
        self.tensor()
    }
}

impl<T: DeviceValue> AsMut<Tensor<T>> for CachedTensor<T> {
    fn as_mut(&mut self) -> &mut Tensor<T> {
        self.tensor_mut()
    }
}

impl<T: DeviceValue> Deref for CachedTensor<T> {
    type Target = Tensor<T>;

    fn deref(&self) -> &Self::Target {
        self.tensor()
    }
}

impl<T: DeviceValue> DerefMut for CachedTensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor_mut()
    }
}

impl<'a, T: DeviceValue> From<&'a CachedTensor<T>> for TensorRef<'a, T> {
    fn from(val: &'a CachedTensor<T>) -> Self {
        val.tensor().as_view()
    }
}

impl<T: DeviceValue> AsTensorRef<T> for CachedTensor<T> {
    fn as_tensor_ref(&self) -> TensorRef<'_, T> {
        (**self).as_tensor_ref()
    }
}

impl<T: DeviceValue> AsTensorRef<T> for &CachedTensor<T> {
    fn as_tensor_ref(&self) -> TensorRef<'_, T> {
        (***self).as_tensor_ref()
    }
}

impl<'a, T: DeviceValue> From<&'a mut CachedTensor<T>> for TensorMut<'a, T> {
    fn from(val: &'a mut CachedTensor<T>) -> Self {
        val.tensor_mut().as_view_mut()
    }
}

impl<T: DeviceValue> AsTensorMut<T> for CachedTensor<T> {
    fn as_tensor_mut(&mut self) -> TensorMut<'_, T> {
        (**self).as_tensor_mut()
    }
}

impl<T: DeviceValue> AsTensorMut<T> for &mut CachedTensor<T> {
    fn as_tensor_mut(&mut self) -> TensorMut<'_, T> {
        (***self).as_tensor_mut()
    }
}

impl<T: DeviceValue> Drop for CachedTensor<T> {
    fn drop(&mut self) {
        if let Some(t) = self.tensor.take() {
            self.cache.reclaim(t, self.usage);
        }
    }
}

pub struct TensorCache {
    // TODO: would be great to have a type-erased tensor type `Tensor<_>` that
    //       we can store in the hashmap. That would avoid the need for `Box<Any>` in
    //       the dashmap.
    #[cfg(not(target_arch = "wasm32"))]
    tensors: Arc<DashMap<TensorKey, Vec<Box<dyn Any + Send + Sync>>>>,
    #[cfg(target_arch = "wasm32")]
    tensors: Arc<DashMap<TensorKey, Vec<Box<dyn Any>>>>,
}

impl Clone for TensorCache {
    fn clone(&self) -> Self {
        Self {
            tensors: self.tensors.clone(),
        }
    }
}

impl Default for TensorCache {
    fn default() -> Self {
        Self {
            tensors: Arc::new(DashMap::new()),
        }
    }
}

impl TensorCache {
    pub fn get<T: DeviceValue>(&self, key: TensorKey) -> Option<CachedTensor<T>> {
        let t = self.tensors.get_mut(&key)?.pop()?;
        let tensor = t
            .downcast()
            .expect("internal error: invalid cached tensor downcast");

        Some(CachedTensor {
            tensor: Some(tensor),
            cache: self.clone(),
            usage: key.usage,
        })
    }

    pub fn get_or_insert<T: DeviceValue>(
        &self,
        key: TensorKey,
        insert: impl FnOnce() -> Result<Tensor<T>, GpuBackendError>,
    ) -> Result<CachedTensor<T>, GpuBackendError> {
        let mut tensors = self.tensors.entry(key).or_default();

        if let Some(t) = tensors.pop() {
            let tensor = t
                .downcast()
                .expect("internal error: invalid cached tensor downcast");
            Ok(CachedTensor {
                tensor: Some(tensor),
                cache: self.clone(),
                usage: key.usage,
            })
        } else {
            let tensor = insert()?;
            Ok(CachedTensor {
                tensor: Some(Box::new(tensor)),
                cache: self.clone(),
                usage: key.usage,
            })
        }
    }

    pub fn enroll<T: DeviceValue>(
        &self,
        tensor: Tensor<T>,
        usage: BufferUsages,
    ) -> CachedTensor<T> {
        CachedTensor {
            tensor: Some(Box::new(tensor)),
            cache: self.clone(),
            usage,
        }
    }

    pub fn clear(&mut self) {
        self.tensors.clear();
    }

    fn reclaim<T: DeviceValue>(&self, t: Box<Tensor<T>>, usage: BufferUsages) {
        let key = TensorKey::new(&t, usage);
        self.tensors.entry(key).or_default().push(t);
    }
}

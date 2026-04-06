use crate::context::LlmContext;
use crate::models::segment_anything::SamModel;
use crate::tensor_cache::CachedTensor;
use khal::backend::GpuBackendError;
use nalgebra::Point2;

pub struct SamPromptEncoderResult {
    pub embd_prompt_sparse: CachedTensor<f32>,
    pub embd_prompt_dense: CachedTensor<f32>,
}

pub fn sam_encode_prompt(
    ctxt: &mut LlmContext,
    model: &SamModel,
    nx: u32,
    ny: u32,
    mut point: Point2<f32>,
) -> Result<SamPromptEncoderResult, GpuBackendError> {
    let hparams = &model.hparams;
    let enc = &model.enc_prompt;

    // transform points
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L276
    {
        let nmax = nx.max(ny);
        let scale = hparams.n_img_size() as f32 / (nmax as f32);
        let nx_new = (nx as f32 * scale + 0.5) as i32;
        let ny_new = (ny as f32 * scale + 0.5) as i32;
        point.x = point.x * (nx_new as f32 / nx as f32) + 0.5;
        point.y = point.y * (ny_new as f32 / ny as f32) + 0.5;
    }

    let inp_data = [
        2.0 * (point.x / hparams.n_img_size() as f32) - 1.0,
        2.0 * (point.y / hparams.n_img_size() as f32) - 1.0,
        // padding
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L81-L85
        2.0 * 0.0 - 1.0,
        2.0 * 0.0 - 1.0,
    ];
    let inp: CachedTensor<f32> = ctxt.tensor([2, 2, 1, 1], &inp_data)?;

    let mut cur = {
        let pe_tr = ctxt.contiguous(enc.pe.as_view().transpose_last_dims())?;
        ctxt.matmul_ggml(&pe_tr, &inp)?
    };

    cur = ctxt.scale(&cur, 2.0 * std::f32::consts::PI)?;

    // concat
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
    {
        let t_sin = ctxt.sin(&cur)?;
        let t_cos = ctxt.cos(&cur)?;

        cur = ctxt.tensor_uninit(&[
            cur.size_ggml(1),
            t_sin.size_ggml(0) + t_cos.size_ggml(0),
            1,
            1,
        ])?;
        let shape = [t_sin.size_ggml(0), t_sin.size_ggml(1)];
        let stride = [Some(1), Some(cur.stride_ggml(1))];
        ctxt.copy(
            t_sin.as_view().squeeze(),
            cur.view_ggml_mut(0, &shape, &stride),
        )?;

        let shape = [t_sin.size_ggml(0), t_sin.size_ggml(1)];
        let stride = [Some(1), Some(cur.stride_ggml(1))];
        ctxt.copy(
            t_cos.as_view().squeeze(),
            cur.view_ggml_mut(t_sin.stride_ggml(1), &shape, &stride),
        )?;

        // overwrite label == -1 with not_a_point_embed.weight
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L86
        // TODO: extend for multiple points
        let offset = cur.stride_ggml(1);
        let shape = [cur.size_ggml(0), 1];
        let stride = [Some(1), Some(cur.stride_ggml(1))];
        ctxt.copy(
            enc.not_a_pt_embd_w.as_view().squeeze(),
            cur.view_ggml_mut(offset, &shape, &stride),
        )?;
    }

    // add point_embeddings[1] to label == 1
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L90
    let shape = [cur.size_ggml(0), 1];
    let stride = [Some(1), Some(cur.stride_ggml(1))];
    let v = cur.view_ggml_mut(0, &shape, &stride);
    ctxt.add_assign(v, &enc.pt_embd[1])?;

    let embd_prompt_sparse = cur;

    let embd_prompt_dense = {
        let w = ctxt.contiguous(enc.no_mask_embd_w.view_ggml(
            0,
            &[1, 1, enc.no_mask_embd_w.size_ggml(0)],
            &[
                Some(1),
                Some(enc.no_mask_embd_w.stride_ggml(0)),
                Some(enc.no_mask_embd_w.stride_ggml(0)),
            ],
        ))?;
        let rep = ctxt.tensor_uninit(&[
            hparams.n_img_embd(),
            hparams.n_img_embd(),
            hparams.n_enc_out_chans,
        ])?;
        ctxt.repeat(&w, &rep)?
    };

    Ok(SamPromptEncoderResult {
        embd_prompt_sparse,
        embd_prompt_dense,
    })
}

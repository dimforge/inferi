use crate::context::LlmContext;
use crate::models::segment_anything::encode_prompt::SamPromptEncoderResult;
use crate::models::segment_anything::layernorm2d::sam_layernorm_2d;
use crate::models::segment_anything::{SamLayerDecTransformerAttn, SamModel, SamState};
use crate::tensor_cache::CachedTensor;
use khal::backend::GpuBackendError;
use vortx::tensor::{AsTensorRef, Tensor};

pub fn sam_decode_mask(
    ctxt: &mut LlmContext,
    model: &SamModel,
    prompt: &SamPromptEncoderResult,
    state: &mut SamState,
) -> Result<(), GpuBackendError> {
    let hparams = &model.hparams;
    let dec = &model.dec;
    let n_img_embd = hparams.n_img_embd();

    let mut tokens;
    {
        // Concatenate output tokens
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L120
        let sparse = &prompt.embd_prompt_sparse;
        tokens = ctxt.tensor_uninit(&[
            dec.iou_token_w.size_ggml(1) + dec.mask_tokens_w.size_ggml(1) + sparse.size_ggml(1),
            dec.iou_token_w.size_ggml(0),
            sparse.size_ggml(2),
        ])?;
        let offsets = [
            0,
            dec.iou_token_w.size_ggml(1) * tokens.stride_ggml(1),
            dec.iou_token_w.size_ggml(1) * tokens.stride_ggml(1)
                + dec.mask_tokens_w.size_ggml(1) * tokens.stride_ggml(1),
        ];

        let shape = [tokens.size_ggml(0), dec.iou_token_w.size_ggml(1)];
        let stride = [Some(1), Some(tokens.stride_ggml(1))];
        ctxt.copy(
            dec.iou_token_w.as_view().squeeze(),
            tokens.view_ggml_mut(offsets[0], &shape, &stride),
        )?;
        let shape = [tokens.size_ggml(0), dec.mask_tokens_w.size_ggml(1)];
        let stride = [Some(1), Some(tokens.stride_ggml(1))];
        ctxt.copy(
            dec.mask_tokens_w.as_view().squeeze(),
            tokens.view_ggml_mut(offsets[1], &shape, &stride),
        )?;
        let shape = [tokens.size_ggml(0), sparse.size_ggml(1)];
        let stride = [Some(1), Some(tokens.stride_ggml(1))];
        ctxt.copy(
            sparse.as_view().squeeze(),
            tokens.view_ggml_mut(offsets[2], &shape, &stride),
        )?;
        // TODO: Sparse prompt embeddings can have more than one point.
    }

    let mut src;
    let mut pos_src;
    let mut src_ne;

    {
        // Expand per-image data in the batch direction to be per-mask
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L125
        src = ctxt.tensor_uninit(&[
            state.embd_img.size_ggml(1),
            state.embd_img.size_ggml(0),
            state.embd_img.size_ggml(2),
            tokens.size_ggml(2),
        ])?;

        src = {
            let rep = ctxt.repeat(&*state.embd_img, &src)?;
            ctxt.add(rep.as_view().squeeze(), &prompt.embd_prompt_dense)?
        };

        src_ne = src.as_view().layout().size;
        src_ne.swap(0, 1); // Convert to ggml convention.

        // flatten & permute
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
        src = ctxt.contiguous(
            src.view_ggml(
                0,
                &[
                    src.size_ggml(0) * src.size_ggml(1),
                    src.size_ggml(2),
                    src.size_ggml(3),
                ],
                &[Some(1), Some(src.stride_ggml(2)), Some(src.stride_ggml(3))],
            )
            .permute_ggml([1, 0, 2, 3]),
        )?;

        pos_src = ctxt.tensor_uninit(&[
            state.pe_img.size_ggml(1),
            state.pe_img.size_ggml(0),
            state.pe_img.size_ggml(2),
            tokens.size_ggml(2),
        ])?;
        pos_src = ctxt.repeat(state.pe_img.as_view().canonicalize(), &pos_src)?;

        // flatten & permute
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
        pos_src = ctxt.contiguous(
            pos_src
                .view_ggml(
                    0,
                    &[
                        pos_src.size_ggml(0) * pos_src.size_ggml(1),
                        pos_src.size_ggml(2),
                        pos_src.size_ggml(3),
                    ],
                    &[
                        Some(1),
                        Some(pos_src.stride_ggml(2)),
                        Some(pos_src.stride_ggml(3)),
                    ],
                )
                .permute_ggml([1, 0, 2, 3]),
        )?;
    }

    let mut queries = ctxt.tensor_uninit(&[0; 4])?; // Will be initialized in the first transformer layer.
    let mut keys = src;

    {
        // Run the transformer
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L62
        for i in 0..model.dec.transformer_layers.len() {
            let tfm_layer = &model.dec.transformer_layers[i];

            // Self attention block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L154
            let skip_first_layer_pe = i == 0;
            if skip_first_layer_pe {
                queries = sam_decode_mask_transformer_attn(
                    ctxt,
                    &tfm_layer.self_attn,
                    &tokens, // queries,
                    &tokens, // queries,
                    &tokens, // queries,
                    model,
                )?;
            } else {
                let q_0 = ctxt.add(queries.as_view().squeeze(), tokens.as_view().squeeze())?;
                let self_attn = sam_decode_mask_transformer_attn(
                    ctxt,
                    &tfm_layer.self_attn,
                    &q_0,
                    &q_0,
                    &queries,
                    model,
                )?;
                queries = ctxt.add(queries.as_view().squeeze(), self_attn.as_view().squeeze())?;
            }

            queries = ctxt.layernorm(&queries, hparams.eps_decoder_transformer)?;
            queries = {
                let w_queries = ctxt.mul(
                    queries.as_view().squeeze(),
                    tfm_layer.norm1_w.as_view().squeeze(),
                )?;
                ctxt.add(
                    w_queries.as_view().squeeze(),
                    tfm_layer.norm1_b.as_view().squeeze(),
                )?
            };

            // Cross attention block, tokens attending to image embedding
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L163
            let q_1 = ctxt.add(queries.as_view().squeeze(), tokens.as_view().squeeze())?;
            let k_1 = ctxt.add(keys.as_view().squeeze(), pos_src.as_view().squeeze())?;

            let cross_attn_token_to_img = sam_decode_mask_transformer_attn(
                ctxt,
                &tfm_layer.cross_attn_token_to_img,
                &q_1,
                &k_1,
                &keys,
                model,
            )?;

            ctxt.add_assign(&mut queries, cross_attn_token_to_img.as_view().squeeze())?;
            queries = ctxt.layernorm(&queries, hparams.eps_decoder_transformer)?;
            queries = {
                let w_queries = ctxt.mul(queries.as_view().squeeze(), &tfm_layer.norm2_w)?;
                ctxt.add(w_queries.as_view().squeeze(), &tfm_layer.norm2_b)?
            };
            // MLP block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L170
            let mut mlp_out = ctxt.matmul_ggml(&tfm_layer.mlp_lin1_w, &queries)?;

            ctxt.add_assign(&mut mlp_out, &tfm_layer.mlp_lin1_b)?;

            // RELU activation
            ctxt.relu_inplace(&mut mlp_out)?;
            mlp_out = ctxt.matmul_ggml(&tfm_layer.mlp_lin2_w, &mlp_out)?;
            ctxt.add_assign(&mut mlp_out, &tfm_layer.mlp_lin2_b)?;

            ctxt.add_assign(&mut queries, &mlp_out)?;
            queries = ctxt.layernorm(&queries, hparams.eps_decoder_transformer)?;
            queries = {
                let w_queries = ctxt.mul(&queries, &tfm_layer.norm3_w)?;
                ctxt.add(&w_queries, &tfm_layer.norm3_b)?
            };

            // Cross attention block, image embedding attending to tokens
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L175
            let q_2 = ctxt.add(&queries, tokens.as_view().squeeze())?;
            let k_2 = ctxt.add(keys.as_view().squeeze(), pos_src.as_view().squeeze())?;
            let cross_attn_img_to_token = sam_decode_mask_transformer_attn(
                ctxt,
                &tfm_layer.cross_attn_img_to_token,
                &k_2,
                &q_2,
                &queries,
                model,
            )?;
            ctxt.add_assign(
                keys.as_view_mut().squeeze(),
                cross_attn_img_to_token.as_view().squeeze(),
            )?;
            keys = ctxt.layernorm(keys.as_view().squeeze(), hparams.eps_decoder_transformer)?;
            keys = {
                let w_keys = ctxt.mul(
                    keys.as_view().squeeze(),
                    tfm_layer.norm4_w.as_view().squeeze(),
                )?;
                ctxt.add(
                    w_keys.as_view().squeeze(),
                    tfm_layer.norm4_b.as_view().squeeze(),
                )?
            };
        }

        // Apply the final attention layer from the points to the image
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L99
        let q = ctxt.add(queries.as_view().squeeze(), tokens.as_view().squeeze())?;
        let k = ctxt.add(keys.as_view().squeeze(), pos_src.as_view().squeeze())?;
        let final_attn_token_to_img = sam_decode_mask_transformer_attn(
            ctxt,
            &dec.transformer_final_attn_token_to_img,
            &q,
            &k,
            &keys,
            model,
        )?;
        ctxt.add_assign(
            queries.as_view_mut().squeeze(),
            final_attn_token_to_img.as_view().squeeze(),
        )?;
        queries = ctxt.layernorm(&queries, hparams.eps_decoder_transformer)?;
        queries = {
            let w_queries = ctxt.mul(&queries, &dec.transformer_norm_final_w)?;
            ctxt.add(
                w_queries.as_view().squeeze(),
                dec.transformer_norm_final_b.as_view().squeeze(),
            )?
        };
    }

    let iou_pred = queries.view_ggml(
        0,
        &[queries.size_ggml(0), queries.size_ggml(2)],
        &[Some(1), Some(queries.stride_ggml(2))],
    );
    let num_mask_tokens = 4; // num_multimask_outputs + 1
    let mask_tokens_out = queries.view_ggml(
        queries.stride_ggml(1),
        &[queries.size_ggml(0), num_mask_tokens, queries.size_ggml(2)],
        &[
            Some(1),
            Some(queries.stride_ggml(1)),
            Some(num_mask_tokens * queries.stride_ggml(1)),
        ],
    );

    // Upscale mask embeddings and predict masks using the mask tokens
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L136
    keys = ctxt.contiguous(keys.as_view().transpose_last_dims())?;

    let keys_view = keys.view_ggml(
        0,
        &src_ne,
        &[
            Some(1),
            Some(src_ne[0] * keys.stride_ggml(0)),
            Some(keys.stride_ggml(1)),
            Some(keys.stride_ggml(2)),
        ],
    );

    let upscaled_embedding;
    {
        // ConvTranspose2d
        // TODO: not 100% sure this runs properly. Some values are not lining up very well
        //       compared to ggml but do look very close.
        keys = ctxt.conv_transpose_2d_p0(&dec.output_upscaling_0_w, keys_view, 2)?;

        {
            let rep = ctxt.repeat(
                dec.output_upscaling_0_b.reshape_ggml(&[
                    1,
                    1,
                    dec.output_upscaling_0_b.size_ggml(0),
                ]),
                keys.as_view().squeeze(),
            )?;
            ctxt.add_assign(keys.as_view_mut().squeeze(), &rep)?;
        }

        keys = sam_layernorm_2d(
            ctxt,
            &keys,
            n_img_embd,
            &dec.output_upscaling_1_w,
            &dec.output_upscaling_1_b,
            hparams.eps,
        )?;

        // GELU activation
        ctxt.gelu_inplace(&mut keys)?;

        // ConvTranspose2d
        keys = ctxt.conv_transpose_2d_p0(&dec.output_upscaling_3_w, &keys, 2)?;
        keys = {
            let rep = ctxt.repeat(
                dec.output_upscaling_3_b.reshape_ggml(&[
                    1,
                    1,
                    dec.output_upscaling_3_b.size_ggml(0),
                    1,
                ]),
                &keys,
            )?;
            ctxt.add(&rep, &keys)?
        };

        // GELU activation
        ctxt.gelu_inplace(&mut keys)?;
        let upscaled_embedding_ = keys.reshape_ggml(&[
            keys.size_ggml(0) * keys.size_ggml(1),
            keys.size_ggml(2),
            keys.size_ggml(3),
        ]);
        // TODO: the transpose shouldn’t be needed
        upscaled_embedding = ctxt.contiguous(upscaled_embedding_.transpose_last_dims())?;
    }

    let mut hyper_in = ctxt.tensor_uninit(&[
        num_mask_tokens,
        n_img_embd / 2,
        mask_tokens_out.size_ggml(2),
    ])?;

    for i in 0..num_mask_tokens {
        let mlp = &dec.output_hypernet_mlps[i as usize];
        let in_ = mask_tokens_out.view_ggml(
            i * mask_tokens_out.stride_ggml(1),
            &[mask_tokens_out.size_ggml(0), mask_tokens_out.size_ggml(2)],
            &[Some(1), Some(mask_tokens_out.stride_ggml(1))],
        );

        let out = sam_decode_mask_mlp_relu_3(
            ctxt, in_, &mlp.w_0, &mlp.b_0, &mlp.w_1, &mlp.b_1, &mlp.w_2, &mlp.b_2,
        )?;

        let shape = [hyper_in.size_ggml(0), hyper_in.size_ggml(2)];
        let stride = [Some(1), Some(hyper_in.stride_ggml(1))];
        let out_offset = i * hyper_in.stride_ggml(1);
        ctxt.copy_with_offsets(
            &out,
            0,
            hyper_in.view_ggml_mut(0, &shape, &stride),
            out_offset,
        )?;
    }

    let masks = ctxt.matmul_ggml(&upscaled_embedding, &hyper_in)?;
    // let mut masks = ctxt.matmul_ggml(&hyper_in, &upscaled_embedding)?;
    // masks = ctxt.contiguous(masks.as_view().transposed())?; // TODO: shouldn’t be needed.
    let masks = masks.reshape_ggml(&[
        keys.size_ggml(0),
        keys.size_ggml(1),
        masks.size_ggml(1),
        keys.size_ggml(3),
    ]);

    // Generate mask quality predictions
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L146
    let iou_pred = sam_decode_mask_mlp_relu_3(
        ctxt,
        iou_pred,
        &dec.iou_prediction_head_0_w,
        &dec.iou_prediction_head_0_b,
        &dec.iou_prediction_head_1_w,
        &dec.iou_prediction_head_1_b,
        &dec.iou_prediction_head_2_w,
        &dec.iou_prediction_head_2_b,
    )?;

    // Select the correct mask or masks for output
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L101
    ctxt.copy_with_offsets(
        iou_pred.view_ggml(0, &[iou_pred.size_ggml(0) - 1, 1], &[Some(1), Some(1)]),
        iou_pred.stride_ggml(0),
        &mut state.iou_predictions,
        0,
    )?;

    let masks = masks.view_ggml(
        masks.stride_ggml(2),
        &[
            masks.size_ggml(0),
            masks.size_ggml(1),
            masks.size_ggml(2) - 1,
            masks.size_ggml(3),
        ],
        &[
            Some(1),
            Some(masks.stride_ggml(1)),
            Some(masks.stride_ggml(2)),
            Some(masks.stride_ggml(3)),
        ],
    );
    state.low_res_masks = ctxt.contiguous(masks)?.into_inner();

    Ok(())
}

fn sam_decode_mask_transformer_attn(
    ctxt: &mut LlmContext,
    attn: &SamLayerDecTransformerAttn,
    queries: &Tensor<f32>,
    keys: &Tensor<f32>,
    values: &Tensor<f32>,
    model: &SamModel,
) -> Result<CachedTensor<f32>, GpuBackendError> {
    let hparams = &model.hparams;
    let n_heads = hparams.n_dec_heads;
    let mut q_cur = ctxt.matmul_ggml(&attn.q_w, queries)?;
    ctxt.add_assign(q_cur.as_view_mut().squeeze(), &attn.q_b)?;

    let mut k_cur = ctxt.matmul_ggml(&attn.k_w, keys)?;
    ctxt.add_assign(k_cur.as_view_mut().squeeze(), &attn.k_b)?;

    let mut v_cur = ctxt.matmul_ggml(&attn.v_w, values)?;
    ctxt.add_assign(v_cur.as_view_mut().squeeze(), &attn.v_b)?;

    let q = q_cur.reshape_ggml(&[
        q_cur.size_ggml(0) / n_heads,
        n_heads,
        q_cur.size_ggml(1),
        q_cur.size_ggml(2),
    ]);
    let q = ctxt.contiguous(q.permute_ggml([0, 2, 1, 3]))?;

    let k = k_cur.reshape_ggml(&[
        k_cur.size_ggml(0) / n_heads,
        n_heads,
        k_cur.size_ggml(1),
        k_cur.size_ggml(2),
    ]);
    let k = ctxt.contiguous(k.permute_ggml([0, 2, 1, 3]))?;

    let v = v_cur.reshape_ggml(&[
        v_cur.size_ggml(0) / n_heads,
        n_heads,
        v_cur.size_ggml(1),
        v_cur.size_ggml(2),
    ]);
    let v = ctxt.contiguous(v.permute_ggml([0, 2, 1, 3]))?;

    // Q * K
    let mut kq = ctxt.matmul_ggml(&k, &q)?;
    ctxt.scale_assign(&mut kq, 1.0 / (q.size_ggml(0) as f32).sqrt())?;
    ctxt.softmax_rows(&mut kq)?;
    let kqv = {
        let v_tr = ctxt.contiguous(v.as_view().transpose_last_dims())?;
        ctxt.matmul_ggml(&kq, &v_tr)?
    };
    let mut kqv_merged = ctxt.contiguous(kqv.as_view().transpose_last_dims())?;
    kqv_merged = ctxt.contiguous(kqv_merged.as_view().permute_ggml([0, 2, 1, 3]))?;
    let kqv_merged = kqv_merged.reshape_ggml(&[
        kqv_merged.size_ggml(0) * kqv_merged.size_ggml(1),
        kqv_merged.size_ggml(2),
        kqv_merged.size_ggml(3),
    ]);
    let mut kqv_merged = ctxt.matmul_ggml(&attn.out_w, kqv_merged)?;
    ctxt.add_assign(kqv_merged.as_view_mut().squeeze(), &attn.out_b)?;

    Ok(kqv_merged)
}

fn sam_decode_mask_mlp_relu_3(
    ctxt: &mut LlmContext,
    in_: impl AsTensorRef<f32>,
    w_0: &Tensor<f32>,
    b_0: &Tensor<f32>,
    w_1: &Tensor<f32>,
    b_1: &Tensor<f32>,
    w_2: &Tensor<f32>,
    b_2: &Tensor<f32>,
) -> Result<CachedTensor<f32>, GpuBackendError> {
    let in_ = in_.as_tensor_ref();
    let mut cur = ctxt.matmul_ggml(w_0, in_)?;
    ctxt.add_assign(&mut cur, b_0)?;
    ctxt.relu_inplace(&mut cur)?;

    let mut cur = ctxt.matmul_ggml(w_1, &cur)?;
    ctxt.add_assign(&mut cur, b_1)?;
    ctxt.relu_inplace(&mut cur)?;

    let mut cur = ctxt.matmul_ggml(w_2, &cur)?;
    ctxt.add_assign(&mut cur, b_2)?;

    Ok(cur)
}

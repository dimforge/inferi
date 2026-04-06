use crate::context::LlmContext;
use crate::models::whisper::model::Whisper;
use crate::models::whisper::WhisperConfig;
use khal::backend::{Backend, GpuBackendError};
use tokenizers::Tokenizer;
use vortx::tensor::Tensor;

const LANGUAGES: [(&str, &str); 99] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
];

/// Returns the token id for the selected language.
pub async fn detect_language(
    ctx: &mut LlmContext<'_>,
    model: &mut Whisper,
    config: &WhisperConfig,
    tokenizer: &Tokenizer,
    mel: &Tensor<f32>,
) -> Result<u32, GpuBackendError> {
    // TODO: avoid the unwraps
    ctx.begin_submission();
    let [_bsize, _, seq_len, _] = mel.layout().size;
    let mel = mel.narrow(2, 0, seq_len.min(config.max_source_positions as u32));
    let mel = ctx.contiguous(mel)?; // TODO PERF: avoid cont
    let language_token_ids = LANGUAGES
        .iter()
        .map(|(t, _)| tokenizer.token_to_id(&format!("<|{t}|>")).unwrap())
        .collect::<Vec<_>>();
    let sot_token = tokenizer.token_to_id(super::SOT_TOKEN).unwrap();
    let audio_features = model.encoder.forward(ctx, &mel, true).await?;
    let tokens = ctx.tensor([1, 1], &[sot_token])?;
    let language_token_ids = ctx.tensor([language_token_ids.len() as u32], &language_token_ids)?;
    let ys = model.decoder.forward(ctx, &tokens, &audio_features, true)?;
    let ys_row = ctx.contiguous(ys.narrow(0, 0, 1))?; // TODO PERF: avoid cont
    let logits = model.decoder.final_linear(ctx, &ys_row)?;
    let logits = logits.as_view().index(0).index(0);
    let mut logits = ctx.select(logits, &language_token_ids, 0)?;
    ctx.softmax_rows(&mut logits)?;
    let probs = logits;
    ctx.submit();

    let probs = ctx.backend.slow_read_vec(probs.buffer()).await?;
    let mut probs = LANGUAGES.iter().zip(probs.iter()).collect::<Vec<_>>();
    probs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for ((_, language), p) in probs.iter().take(5) {
        println!("{language}: {p}")
    }
    let language = tokenizer
        .token_to_id(&format!("<|{}|>", probs[0].0 .0))
        .unwrap();
    Ok(language)
}

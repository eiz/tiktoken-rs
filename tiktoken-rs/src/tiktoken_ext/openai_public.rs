pub const ENDOFTEXT: &str = "<|endoftext|>";
pub const FIM_PREFIX: &str = "<|fim_prefix|>";
pub const FIM_MIDDLE: &str = "<|fim_middle|>";
pub const FIM_SUFFIX: &str = "<|fim_suffix|>";
pub const ENDOFPROMPT: &str = "<|endofprompt|>";
pub const STARTOFTRANSCRIPT: &str = "<|startoftranscript|>";
pub const TRANSLATE: &str = "<|translate|>";
pub const TRANSCRIBE: &str = "<|transcribe|>";
pub const STARTOFLM: &str = "<|startoflm|>";
pub const STARTOFPREV: &str = "<|startofprev|>";
pub const NOSPEECH: &str = "<|nospeech|>";
pub const NOTIMESTAMPS: &str = "<|notimestamps|>";
pub const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su",
];

/// Adaptation of the tiktoken crate for use in Rust projects
use anyhow::Result;
use base64::{engine::general_purpose, Engine as _};

use rustc_hash::FxHashMap as HashMap;

use crate::vendor_tiktoken::CoreBPE;

const R50K_BASE_BPE_FILE: &str = include_str!("../../assets/r50k_base.tiktoken");

/// Use for GPT-3 models like `davinci`
/// Initializes and returns a new instance of the r50k_base tokenizer (also known as `gpt2`)
pub fn r50k_base() -> Result<CoreBPE> {
    let mut encoder = HashMap::default();
    for line in R50K_BASE_BPE_FILE.lines() {
        let mut parts = line.split(' ');
        let token = &general_purpose::STANDARD.decode(parts.next().unwrap())?;
        let rank: usize = parts.next().unwrap().parse().unwrap();
        encoder.insert(token.clone(), rank);
    }

    let mut special_tokens = HashMap::default();
    special_tokens.insert(String::from(ENDOFTEXT), 50256);

    let bpe = CoreBPE::new(
        encoder,
        special_tokens,
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    )?;
    Ok(bpe)
}

fn add_special_token(
    special_tokens: &mut HashMap<String, usize>,
    token_index: &mut usize,
    token: &str,
) {
    special_tokens.insert(String::from(token), *token_index);
    *token_index += 1;
}

fn get_whisper_special_tokens(
    encoder: &std::collections::HashMap<
        Vec<u8>,
        usize,
        std::hash::BuildHasherDefault<rustc_hash::FxHasher>,
    >,
) -> std::collections::HashMap<String, usize, std::hash::BuildHasherDefault<rustc_hash::FxHasher>> {
    let mut special_tokens = HashMap::default();
    let mut token_index = encoder.len();
    add_special_token(&mut special_tokens, &mut token_index, ENDOFTEXT);
    add_special_token(&mut special_tokens, &mut token_index, STARTOFTRANSCRIPT);

    for lang in LANGUAGES {
        add_special_token(
            &mut special_tokens,
            &mut token_index,
            &format!("<|{}|>", lang),
        );
    }

    add_special_token(&mut special_tokens, &mut token_index, TRANSLATE);
    add_special_token(&mut special_tokens, &mut token_index, TRANSCRIBE);
    add_special_token(&mut special_tokens, &mut token_index, STARTOFLM);
    add_special_token(&mut special_tokens, &mut token_index, STARTOFPREV);
    add_special_token(&mut special_tokens, &mut token_index, NOSPEECH);
    add_special_token(&mut special_tokens, &mut token_index, NOTIMESTAMPS);

    for i in 0..=1500 {
        add_special_token(
            &mut special_tokens,
            &mut token_index,
            &format!("<|{:.2}|>", i as f32 * 0.02),
        );
    }
    special_tokens
}

/// Use for OpenAI `whisper` speech to text model (gpt2 version)
/// Initializes and returns a new instance of the whisper tokenizer.
pub fn whisper_gpt2() -> Result<CoreBPE> {
    let mut encoder = HashMap::default();
    for line in R50K_BASE_BPE_FILE.lines() {
        let mut parts = line.split(' ');
        let token = &general_purpose::STANDARD.decode(parts.next().unwrap())?;
        let rank: usize = parts.next().unwrap().parse().unwrap();
        encoder.insert(token.clone(), rank);
    }

    let special_tokens = get_whisper_special_tokens(&encoder);

    let bpe = CoreBPE::new(
        encoder,
        special_tokens,
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    )?;
    Ok(bpe)
}

/// Use for OpenAI `whisper` speech to text model (multilingual version)
/// Initializes and returns a new instance of the whisper tokenizer.
pub fn whisper_multilingual() -> Result<CoreBPE> {
    let bpe_file = include_str!("../../assets/whisper_multilingual.tiktoken");

    let mut encoder = HashMap::default();
    for line in bpe_file.lines() {
        let mut parts = line.split(' ');
        let token_b64 = parts.next().unwrap();
        // multilingual model includes an empty token represented as '=' which
        // python's base64 will parse but other libraries will not
        let token = &if token_b64 == "=" {
            vec![]
        } else {
            match general_purpose::STANDARD.decode(token_b64) {
                Ok(v) => v,
                Err(e) => panic!("bad shit {:?} {:?}", token_b64, e),
            }
        };
        let rank: usize = parts.next().unwrap().parse().unwrap();
        encoder.insert(token.clone(), rank);
    }

    let special_tokens = get_whisper_special_tokens(&encoder);

    let bpe = CoreBPE::new(
        encoder,
        special_tokens,
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    )?;
    Ok(bpe)
}

/// Use for Code models, `text-davinci-002`, `text-davinci-003`
/// Initializes and returns a new instance of the p50k_base tokenizer.
pub fn p50k_base() -> Result<CoreBPE> {
    let bpe_file = include_str!("../../assets/p50k_base.tiktoken");

    let mut encoder = HashMap::default();
    for line in bpe_file.lines() {
        let mut parts = line.split(' ');
        let raw = parts.next().unwrap();
        let token = &general_purpose::STANDARD.decode(raw)?;
        let rank: usize = parts.next().unwrap().parse().unwrap();
        encoder.insert(token.clone(), rank);
    }

    let mut special_tokens = HashMap::default();
    special_tokens.insert(String::from(ENDOFTEXT), 50256);

    let bpe = CoreBPE::new(
        encoder,
        special_tokens,
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    )?;
    Ok(bpe)
}

/// Use for edit models like `text-davinci-edit-001`, `code-davinci-edit-001`
/// Initializes and returns a new instance of the p50k_base tokenizer.
pub fn p50k_edit() -> Result<CoreBPE> {
    let bpe_file = include_str!("../../assets/p50k_base.tiktoken");

    let mut encoder = HashMap::default();
    for line in bpe_file.lines() {
        let mut parts = line.split(' ');
        let raw = parts.next().unwrap();
        let token = &general_purpose::STANDARD.decode(raw)?;
        let rank: usize = parts.next().unwrap().parse().unwrap();
        encoder.insert(token.clone(), rank);
    }

    let mut special_tokens = HashMap::default();
    special_tokens.insert(String::from(ENDOFTEXT), 50256);
    special_tokens.insert(String::from(FIM_PREFIX), 50281);
    special_tokens.insert(String::from(FIM_MIDDLE), 50282);
    special_tokens.insert(String::from(FIM_SUFFIX), 50283);

    let bpe = CoreBPE::new(
        encoder,
        special_tokens,
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    )?;
    Ok(bpe)
}

/// Use for ChatGPT models, `text-embedding-ada-002`
/// Initializes and returns a new instance of the cl100k_base tokenizer.
pub fn cl100k_base() -> Result<CoreBPE> {
    let cl100k_base = include_str!("../../assets/cl100k_base.tiktoken");

    let mut encoder = HashMap::default();
    for line in cl100k_base.lines() {
        let mut parts = line.split(' ');
        let raw = parts.next().unwrap();
        let token = &general_purpose::STANDARD.decode(raw)?;
        let rank: usize = parts.next().unwrap().parse().unwrap();
        encoder.insert(token.clone(), rank);
    }

    let mut special_tokens = HashMap::default();
    special_tokens.insert(String::from(ENDOFTEXT), 100257);
    special_tokens.insert(String::from(FIM_PREFIX), 100258);
    special_tokens.insert(String::from(FIM_MIDDLE), 100259);
    special_tokens.insert(String::from(FIM_SUFFIX), 100260);
    special_tokens.insert(String::from(ENDOFPROMPT), 100276);

    let bpe = CoreBPE::new(
        encoder,
        special_tokens,
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    )?;
    Ok(bpe)
}

//! Safe wrapper around multimodal (MTMD) functionality in llama.cpp.
//!
//! This module provides Rust bindings for llama.cpp's multimodal support,
//! allowing processing of text, image, and audio inputs through a unified interface.
//!
//! # Warning
//! This API is experimental and subject to breaking changes.
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::slice;

use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::ptr::Ptr;
use crate::token::LlamaToken;

/// Input chunk types for multimodal data
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::MtmdInputChunkType;
///
/// let text_chunk = MtmdInputChunkType::Text;
/// let image_chunk = MtmdInputChunkType::Image;
/// let audio_chunk = MtmdInputChunkType::Audio;
///
/// assert_eq!(text_chunk, MtmdInputChunkType::Text);
/// assert_eq!(text_chunk, llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_TEXT.into());
/// assert_ne!(text_chunk, image_chunk);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MtmdInputChunkType {
    /// Text input chunk
    Text = llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_TEXT as _,
    /// Image input chunk
    Image = llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_IMAGE as _,
    /// Audio input chunk
    Audio = llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_AUDIO as _,
}

impl From<llama_cpp_sys_2::mtmd_input_chunk_type> for MtmdInputChunkType {
    fn from(chunk_type: llama_cpp_sys_2::mtmd_input_chunk_type) -> Self {
        match chunk_type {
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_TEXT => MtmdInputChunkType::Text,
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_IMAGE => MtmdInputChunkType::Image,
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_AUDIO => MtmdInputChunkType::Audio,
            _ => panic!("Unknown MTMD input chunk type: {chunk_type}"),
        }
    }
}

/// Configuration parameters for MTMD context
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::{MtmdContextParams, mtmd_default_marker};
/// use std::ffi::CString;
///
/// let params = MtmdContextParams {
///     use_gpu: false,
///     print_timings: true,
///     n_threads: 4,
///     media_marker: CString::new(mtmd_default_marker()).unwrap(),
///     image_min_tokens: -1,
///     image_max_tokens: -1,
///     ..MtmdContextParams::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MtmdContextParams {
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Whether to print timing information
    pub print_timings: bool,
    /// Number of threads to use for processing
    pub n_threads: i32,
    /// Media marker string used to identify media positions in text
    pub media_marker: CString,
    /// Flash attention policy for the multimodal encoder
    pub flash_attn_type: llama_cpp_sys_2::llama_flash_attn_type,
    /// Whether to run a warmup encode pass during initialization
    pub warmup: bool,
    /// Minimum number of tokens used to represent an image.
    /// Controls the visual token budget lower bound. Use -1 for the model default.
    /// Gemma 4 supported budgets: 70, 140, 280, 560, 1120.
    pub image_min_tokens: i32,
    /// Maximum number of tokens used to represent an image.
    /// Controls the visual token budget upper bound. Use -1 for the model default.
    /// Lower values reduce memory and compute at the cost of visual detail.
    /// Gemma 4 supported budgets: 70, 140, 280, 560, 1120.
    pub image_max_tokens: i32,
    /// Maximum number of output tokens in one `MtmdBatch` encode pass (default: 1024).
    pub batch_max_tokens: i32,
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        unsafe { llama_cpp_sys_2::mtmd_context_params_default() }.into()
    }
}

impl From<&MtmdContextParams> for llama_cpp_sys_2::mtmd_context_params {
    fn from(params: &MtmdContextParams) -> Self {
        let mut context = unsafe { llama_cpp_sys_2::mtmd_context_params_default() };
        let MtmdContextParams {
            use_gpu,
            print_timings,
            n_threads,
            media_marker,
            flash_attn_type,
            warmup,
            image_min_tokens,
            image_max_tokens,
            batch_max_tokens,
        } = params;

        context.use_gpu = *use_gpu;
        context.print_timings = *print_timings;
        context.n_threads = *n_threads;
        context.media_marker = media_marker.as_ptr();
        context.flash_attn_type = *flash_attn_type;
        context.warmup = *warmup;
        context.image_min_tokens = *image_min_tokens;
        context.image_max_tokens = *image_max_tokens;
        context.batch_max_tokens = *batch_max_tokens;

        context
    }
}

impl From<llama_cpp_sys_2::mtmd_context_params> for MtmdContextParams {
    fn from(params: llama_cpp_sys_2::mtmd_context_params) -> Self {
        Self {
            use_gpu: params.use_gpu,
            print_timings: params.print_timings,
            n_threads: params.n_threads,
            media_marker: unsafe { CStr::from_ptr(params.media_marker) }.to_owned(),
            flash_attn_type: params.flash_attn_type,
            warmup: params.warmup,
            image_min_tokens: params.image_min_tokens,
            image_max_tokens: params.image_max_tokens,
            batch_max_tokens: params.batch_max_tokens,
        }
    }
}

/// Capabilities reported by an `mmproj` file without initializing the full context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtmdCaps {
    /// Whether the projector supports vision input.
    pub inp_vision: bool,
    /// Whether the projector supports audio input.
    pub inp_audio: bool,
    /// Whether the projector supports audio generation.
    pub gen_audio: bool,
}

/// Inspects `mmproj_path` capabilities without loading weights into a full context.
///
/// # Errors
/// Returns `MtmdInitError::CStringError` when `mmproj_path` contains a NUL byte.
pub fn mtmd_get_cap_from_file(mmproj_path: &str) -> Result<MtmdCaps, MtmdInitError> {
    let path_cstr = CString::new(mmproj_path)?;
    let caps = unsafe { llama_cpp_sys_2::mtmd_get_cap_from_file(path_cstr.as_ptr()) };
    Ok(MtmdCaps {
        inp_vision: caps.inp_vision,
        inp_audio: caps.inp_audio,
        gen_audio: caps.gen_audio,
    })
}

/// Text input configuration
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::MtmdInputText;
///
/// let input = MtmdInputText {
///     text: "Describe this image.".to_string(),
///     add_special: true,
///     parse_special: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MtmdInputText {
    /// The input text string
    pub text: String,
    /// Whether to add special tokens
    pub add_special: bool,
    /// Whether to parse special tokens
    pub parse_special: bool,
}

/// Boxed cancellation predicate retained for the lifetime of an [`MtmdContext`].
struct EvalAbortState {
    should_abort: Box<dyn Fn() -> bool + Send + Sync>,
}

unsafe extern "C" fn mtmd_progress_trampoline<F>(
    progress: f32,
    user_data: *mut std::ffi::c_void,
) -> bool
where
    F: FnMut(f32) -> bool,
{
    if user_data.is_null() {
        return true;
    }
    let callback = &mut *user_data.cast::<F>();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(progress))).unwrap_or(false)
}

unsafe extern "C" fn mtmd_eval_abort_trampoline(
    _tensor: *mut llama_cpp_sys_2::ggml_tensor,
    ask: bool,
    user_data: *mut std::ffi::c_void,
) -> bool {
    if user_data.is_null() {
        return !ask;
    }
    let state = &*user_data.cast::<EvalAbortState>();
    let aborted =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (state.should_abort)()))
            .unwrap_or(true);
    if ask {
        // Return `false` while running so the scheduler batches nodes at full speed; return
        // `true` once cancelled so the scheduler invokes us with `ask == false` to cancel.
        aborted
    } else {
        !aborted
    }
}

/// Safe wrapper around `mtmd_context`.
///
/// This represents an initialized multimodal context that can process
/// text, images, and audio through llama.cpp's multimodal interface.
pub struct MtmdContext {
    context: Ptr<llama_cpp_sys_2::mtmd_context>,
    // Retained until after `mtmd_free` runs in `Drop` because `mtmd_backends` holds its raw pointer.
    _eval_abort: Option<Box<EvalAbortState>>,
}

impl std::fmt::Debug for MtmdContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtmdContext")
            .field("context", &self.context)
            .field("has_eval_abort", &self._eval_abort.is_some())
            .finish()
    }
}

// SAFETY: `mtmd_context` tokenization is thread safe.
//
// Note that evaluation and encoding is not thread-safe, but we ensure this
// doesn't happen on multiple threads by making those methods take `&mut`.
unsafe impl Send for MtmdContext {}
// SAFETY: Same as above.
// Unlike `LlamaContext`, this context doesn't need to synchronize at various
// points, so there's no danger there.
unsafe impl Sync for MtmdContext {}

impl MtmdContext {
    /// Initialize MTMD context from a multimodal projection file.
    ///
    /// # Arguments
    ///
    /// * `mmproj_path` - Path to the multimodal projection file
    /// * `text_model` - Reference to the text model
    /// * `params` - Configuration parameters for the MTMD context
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdContext)` on success, or `Err(MtmdInitError)` on failure.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The path cannot be converted to a C string
    /// - The underlying C function returns null (indicating initialization failure)
    pub fn init_from_file(
        mmproj_path: &str,
        text_model: &LlamaModel,
        params: &MtmdContextParams,
    ) -> Result<Self, MtmdInitError> {
        let path_cstr = CString::new(mmproj_path)?;
        let ctx_params = llama_cpp_sys_2::mtmd_context_params::from(params);

        let context = unsafe {
            llama_cpp_sys_2::mtmd_init_from_file(
                path_cstr.as_ptr(),
                text_model.model.as_ptr(),
                ctx_params,
            )
        };

        let context = Ptr::new(context).ok_or(MtmdInitError::NullResult)?;
        Ok(Self {
            context,
            _eval_abort: None,
        })
    }

    /// Initialize MTMD context with optional load-progress and encode-cancellation callbacks.
    ///
    /// `progress_callback` receives progress in `[0.0, 1.0]` during `mtmd_init_from_file` and
    /// aborts loading when it returns `false`. `should_abort_eval` is queried during vision graph
    /// evaluation and cancels encoding when it returns `true`.
    ///
    /// # Errors
    /// Returns `MtmdInitError::CStringError` if `mmproj_path` contains a NUL byte, or
    /// `MtmdInitError::NullResult` if initialization fails or is aborted.
    pub fn init_from_file_with_callbacks<P, A>(
        mmproj_path: &str,
        text_model: &LlamaModel,
        params: &MtmdContextParams,
        mut progress_callback: Option<P>,
        should_abort_eval: Option<A>,
    ) -> Result<Self, MtmdInitError>
    where
        P: FnMut(f32) -> bool,
        A: Fn() -> bool + Send + Sync + 'static,
    {
        let path_cstr = CString::new(mmproj_path)?;
        let mut ctx_params = llama_cpp_sys_2::mtmd_context_params::from(params);
        if let Some(callback) = progress_callback.as_mut() {
            ctx_params.progress_callback = Some(mtmd_progress_trampoline::<P>);
            ctx_params.progress_callback_user_data = std::ptr::from_mut(callback).cast();
        }
        let mut eval_abort = should_abort_eval.map(|should_abort| {
            Box::new(EvalAbortState {
                should_abort: Box::new(should_abort),
            })
        });
        if let Some(state) = eval_abort.as_mut() {
            ctx_params.cb_eval = Some(mtmd_eval_abort_trampoline);
            ctx_params.cb_eval_user_data = std::ptr::from_mut(state.as_mut()).cast();
        }

        let context = unsafe {
            llama_cpp_sys_2::mtmd_init_from_file(
                path_cstr.as_ptr(),
                text_model.model.as_ptr(),
                ctx_params,
            )
        };

        let context = Ptr::new(context).ok_or(MtmdInitError::NullResult)?;
        Ok(Self {
            context,
            _eval_abort: eval_abort,
        })
    }

    /// Check whether non-causal attention mask is needed before `llama_decode`.
    #[must_use]
    pub fn decode_use_non_causal(&self) -> bool {
        unsafe {
            llama_cpp_sys_2::mtmd_decode_use_non_causal(self.context.as_ptr(), std::ptr::null())
        }
    }

    /// Check whether the current model uses M-RoPE for `llama_decode`.
    ///
    /// M-RoPE (Multimodal Rotary Position Embedding) affects how positions
    /// are calculated for multimodal inputs.
    #[must_use]
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_decode_use_mrope(self.context.as_ptr()) }
    }

    /// Check whether the current model supports vision input.
    #[must_use]
    pub fn support_vision(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_support_vision(self.context.as_ptr()) }
    }

    /// Check whether the current model supports audio input.
    #[must_use]
    pub fn support_audio(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_support_audio(self.context.as_ptr()) }
    }

    /// Get audio sample rate in Hz (e.g., 16000 for Whisper).
    /// Returns None if audio is not supported.
    #[must_use]
    pub fn get_audio_sample_rate(&self) -> Option<u32> {
        let rate = unsafe { llama_cpp_sys_2::mtmd_get_audio_sample_rate(self.context.as_ptr()) };
        (rate > 0).then_some(rate.unsigned_abs())
    }

    /// Backward-compatible alias for the audio sample rate getter.
    #[must_use]
    pub fn get_audio_bitrate(&self) -> Option<u32> {
        self.get_audio_sample_rate()
    }

    /// Tokenize input text and bitmaps into chunks.
    ///
    /// The input text must contain media markers (default: `<__media__>`) that will be
    /// replaced with the corresponding bitmap data from the `bitmaps` array.
    /// The number of bitmaps must equal the number of markers in the text.
    ///
    /// # Arguments
    ///
    /// * `text` - Text input configuration containing the text and tokenization options
    /// * `bitmaps` - Array of bitmaps (images/audio) to replace markers with
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdInputChunks)` containing the tokenized chunks on success.
    ///
    /// # Errors
    ///
    /// * `BitmapCountMismatch` - Number of bitmaps doesn't match number of markers
    /// * `ImagePreprocessingError` - Error occurred during image preprocessing
    /// * `UnknownError` - Other tokenization error occurred
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use llama_cpp_2::mtmd::*;
    /// # fn example(ctx: &MtmdContext, bitmap: &MtmdBitmap) -> Result<(), Box<dyn std::error::Error>> {
    /// let text = MtmdInputText {
    ///     text: "Here is an image: <__media__>\nDescribe it.".to_string(),
    ///     add_special: true,
    ///     parse_special: true,
    /// };
    /// let chunks = ctx.tokenize(text, &[bitmap])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn tokenize(
        &self,
        text: MtmdInputText,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<MtmdInputChunks, MtmdTokenizeError> {
        let mut chunks = MtmdInputChunks::new();
        let text_cstring = CString::new(text.text)?;
        let input_text = llama_cpp_sys_2::mtmd_input_text {
            text: text_cstring.as_ptr(),
            text_len: text_cstring.as_bytes().len(),
            add_special: text.add_special,
            parse_special: text.parse_special,
        };

        // Create bitmap pointers
        let bitmap_ptrs: Vec<*const llama_cpp_sys_2::mtmd_bitmap> =
            bitmaps.iter().map(|b| b.bitmap.as_ptr()).collect();

        let result = unsafe {
            llama_cpp_sys_2::mtmd_tokenize(
                self.context.as_ptr(),
                chunks.chunks.as_mut_ptr(),
                &raw const input_text,
                bitmap_ptrs.as_ptr().cast_mut(),
                bitmaps.len(),
            )
        };

        match result {
            0 => Ok(chunks),
            1 => Err(MtmdTokenizeError::BitmapCountMismatch),
            2 => Err(MtmdTokenizeError::ImagePreprocessingError),
            _ => Err(MtmdTokenizeError::UnknownError(result)),
        }
    }

    /// Encode a chunk for image/audio processing.
    ///
    /// This function processes image or audio chunks by encoding them into
    /// embeddings that can be used by the language model. The embeddings
    /// can be retrieved using `get_output_embeddings()`.
    ///
    /// # Arguments
    ///
    /// * `chunk` - The input chunk to encode (should be image or audio type)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEncodeError::EncodeFailure` if encoding fails.
    pub fn encode_chunk(&mut self, chunk: &MtmdInputChunk<'_>) -> Result<(), MtmdEncodeError> {
        let result = unsafe {
            llama_cpp_sys_2::mtmd_encode_chunk(self.context.as_mut_ptr(), chunk.chunk.as_ptr())
        };

        if result == 0 {
            Ok(())
        } else {
            Err(MtmdEncodeError::EncodeFailure(result))
        }
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::mtmd_free(self.context.as_mut_ptr()) }
    }
}

/// Safe wrapper around `mtmd_bitmap`.
///
/// Represents bitmap data for images or audio that can be processed
/// by the multimodal system. For images, data is stored in RGB format.
/// For audio, data is stored as PCM F32 samples.
#[derive(Debug)]
pub struct MtmdBitmap {
    bitmap: Ptr<llama_cpp_sys_2::mtmd_bitmap>,
}

// SAFETY: `mtmd_bitmap` is thread safe.
unsafe impl Send for MtmdBitmap {}
// SAFETY: Same as above.
unsafe impl Sync for MtmdBitmap {}

impl MtmdBitmap {
    /// Create a bitmap from image data in RGB format.
    ///
    /// # Arguments
    ///
    /// * `nx` - Width of the image in pixels
    /// * `ny` - Height of the image in pixels
    /// * `data` - Image data in RGBRGBRGB... format (must be exactly `nx * ny * 3` bytes)
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `InvalidDataSize` - Data length doesn't match `nx * ny * 3`
    /// * `NullResult` - Underlying C function returned null
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdBitmap;
    ///
    /// // Create a 2x2 red image
    /// let red_pixel = [255, 0, 0]; // RGB values for red
    /// let image_data = red_pixel.repeat(4); // 2x2 = 4 pixels
    ///
    /// let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data);
    /// assert!(bitmap.is_ok());
    /// ```
    pub fn from_image_data(nx: u32, ny: u32, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        if data.len() != (nx * ny * 3) as usize {
            return Err(MtmdBitmapError::InvalidDataSize);
        }

        let bitmap = unsafe { llama_cpp_sys_2::mtmd_bitmap_init(nx, ny, data.as_ptr()) };

        let bitmap = Ptr::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from audio data in PCM F32 format.
    ///
    /// # Arguments
    ///
    /// * `data` - Audio samples as 32-bit floating point values
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `NullResult` - Underlying C function returned null
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdBitmap;
    ///
    /// // Create a simple sine wave audio sample
    /// let audio_data: Vec<f32> = (0..100)
    ///     .map(|i| (i as f32 * 0.1).sin())
    ///     .collect();
    ///
    /// let bitmap = MtmdBitmap::from_audio_data(&audio_data);
    /// // Note: This will likely fail without proper MTMD context setup
    /// ```
    pub fn from_audio_data(data: &[f32]) -> Result<Self, MtmdBitmapError> {
        let bitmap =
            unsafe { llama_cpp_sys_2::mtmd_bitmap_init_from_audio(data.len(), data.as_ptr()) };

        let bitmap = Ptr::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from a file.
    ///
    /// Supported formats:
    /// - Images: formats supported by `stb_image` (jpg, png, bmp, gif, etc.)
    /// - Audio: formats supported by miniaudio (wav, mp3, flac)
    ///
    /// Audio files are auto-detected based on magic bytes.
    ///
    /// # Arguments
    ///
    /// * `ctx` - MTMD context for processing
    /// * `path` - Path to the image or audio file
    /// * `placeholder` - If `true`, build a data-less bitmap (dimensions/length only, with no
    ///   decoded pixels or audio samples) — useful for counting tokens without loading the media.
    ///   If `false`, decode and load the actual data.
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `CStringError` - Path contains null bytes
    /// * `NullResult` - File could not be loaded or processed
    ///
    /// This function is thread-safe.
    pub fn from_file(
        ctx: &MtmdContext,
        path: &str,
        placeholder: bool,
    ) -> Result<Self, MtmdBitmapError> {
        let path_cstr = CString::new(path)?;
        // This helper now returns a wrapper struct (bitmap + an optional video
        // decoding context) instead of a bare pointer. `video_ctx` is only set
        // when MTMD_VIDEO is enabled at compile time; this crate does not build
        // with it on, so it is always null here and only the bitmap matters.
        let wrapper = unsafe {
            llama_cpp_sys_2::mtmd_helper_bitmap_init_from_file(
                ctx.context.as_ptr(),
                path_cstr.as_ptr(),
                placeholder,
                llama_cpp_sys_2::mtmd_helper_init_opt_default(),
            )
        };

        let bitmap = Ptr::new(wrapper.bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from a buffer containing file data.
    ///
    /// Supported formats:
    /// - Images: formats supported by `stb_image` (jpg, png, bmp, gif, etc.)
    /// - Audio: formats supported by miniaudio (wav, mp3, flac)
    ///
    /// Audio files are auto-detected based on magic bytes.
    ///
    /// # Arguments
    ///
    /// * `ctx` - MTMD context for processing
    /// * `data` - Buffer containing the file data
    /// * `placeholder` - If `true`, build a data-less bitmap (dimensions/length only, with no
    ///   decoded pixels or audio samples) — useful for counting tokens without loading the media.
    ///   If `false`, decode and load the actual data.
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `NullResult` - Buffer could not be processed
    ///
    /// This function is thread-safe.
    pub fn from_buffer(
        ctx: &MtmdContext,
        data: &[u8],
        placeholder: bool,
    ) -> Result<Self, MtmdBitmapError> {
        // See the comment in `from_file`: this returns a wrapper struct now, and
        // `video_ctx` is always null since MTMD_VIDEO is not enabled in this build.
        let wrapper = unsafe {
            llama_cpp_sys_2::mtmd_helper_bitmap_init_from_buf(
                ctx.context.as_ptr(),
                data.as_ptr(),
                data.len(),
                placeholder,
                llama_cpp_sys_2::mtmd_helper_init_opt_default(),
            )
        };

        let bitmap = Ptr::new(wrapper.bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Get bitmap width in pixels.
    #[must_use]
    pub fn nx(&self) -> u32 {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_get_nx(self.bitmap.as_ptr()) }
    }

    /// Get bitmap height in pixels.
    #[must_use]
    pub fn ny(&self) -> u32 {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_get_ny(self.bitmap.as_ptr()) }
    }

    /// Get bitmap data as a byte slice.
    ///
    /// For images: RGB format with length `nx * ny * 3`
    /// For audio: PCM F32 format with length `n_samples * 4`
    #[must_use]
    pub fn data(&self) -> &[u8] {
        let ptr = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_data(self.bitmap.as_ptr()) };
        let len = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_n_bytes(self.bitmap.as_ptr()) };
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    /// Check if this bitmap contains audio data (vs image data).
    #[must_use]
    pub fn is_audio(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_is_audio(self.bitmap.as_ptr()) }
    }

    /// Check if this bitmap contains voice data.
    #[must_use]
    pub fn is_voice(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_is_voice(self.bitmap.as_ptr()) }
    }

    /// Get the bitmap's optional ID string.
    ///
    /// Bitmap ID is useful for KV cache tracking and can e.g. be calculated
    /// based on a hash of the bitmap data.
    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_id(self.bitmap.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            let id = unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned();
            Some(id)
        }
    }

    /// Set the bitmap's ID string.
    ///
    /// Bitmap ID is useful for KV cache tracking and can e.g. be calculated
    /// based on a hash of the bitmap data.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID string to set
    ///
    /// # Errors
    ///
    /// Returns an error if the ID string contains null bytes.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use llama_cpp_2::mtmd::MtmdBitmap;
    /// # fn example(bitmap: &mut MtmdBitmap) -> Result<(), Box<dyn std::error::Error>> {
    /// bitmap.set_id("image_001")?;
    /// assert_eq!(bitmap.id(), Some("image_001".to_string()));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_id(&mut self, id: &str) -> Result<(), std::ffi::NulError> {
        let id_cstr = CString::new(id)?;
        unsafe {
            llama_cpp_sys_2::mtmd_bitmap_set_id(self.bitmap.as_mut_ptr(), id_cstr.as_ptr());
        }
        Ok(())
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_free(self.bitmap.as_mut_ptr()) }
    }
}

/// Safe wrapper around `mtmd_input_chunks`.
///
/// This is a collection of input chunks created from tokenizing text and media.
/// The chunks represent the tokenized input that can be processed by the model,
/// with text chunks containing tokens and media chunks containing embeddings.
#[derive(Debug)]
pub struct MtmdInputChunks {
    chunks: Ptr<llama_cpp_sys_2::mtmd_input_chunks>,
}

// SAFETY: `mtmd_input_chunks` is thread-safe, it's just a list of chunks.
unsafe impl Send for MtmdInputChunks {}
// SAFETY: Same as above.
unsafe impl Sync for MtmdInputChunks {}

impl Default for MtmdInputChunks {
    fn default() -> Self {
        Self::new()
    }
}

impl MtmdInputChunks {
    /// Create a new empty input chunks collection
    /// # Panics
    /// This function will panic if the underlying llama.cpp function returns null,
    /// which should not happen.
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdInputChunks;
    ///
    /// let chunks = MtmdInputChunks::new();
    /// assert_eq!(chunks.len(), 0);
    /// assert!(chunks.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        let chunks = unsafe { llama_cpp_sys_2::mtmd_input_chunks_init() };
        let chunks = Ptr::new(chunks).unwrap();
        Self { chunks }
    }

    /// Get the number of chunks
    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { llama_cpp_sys_2::mtmd_input_chunks_size(self.chunks.as_ptr()) }
    }

    /// Check if chunks collection is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a chunk by index.
    ///
    /// The returned chunk borrows from this collection. Dropping the collection
    /// while a borrowed chunk is still in use is a compile error. Use
    /// [`MtmdInputChunk::copy`] if you need a chunk that outlives the collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdInputChunks;
    ///
    /// let chunks = MtmdInputChunks::new();
    /// assert!(chunks.get(0).is_none());
    /// ```
    ///
    /// ```compile_fail
    /// use llama_cpp_2::mtmd::MtmdInputChunks;
    ///
    /// let chunks = MtmdInputChunks::new();
    /// let chunk = chunks.get(0);
    /// drop(chunks);
    /// let _ = chunk;
    /// ```
    #[must_use]
    pub fn get(&self, index: usize) -> Option<MtmdInputChunk<'_>> {
        if index >= self.len() {
            return None;
        }

        let chunk_ptr =
            unsafe { llama_cpp_sys_2::mtmd_input_chunks_get(self.chunks.as_ptr(), index) };

        // Note: We don't own this chunk, it's owned by the chunks collection
        Ptr::new(chunk_ptr.cast_mut()).map(|ptr| MtmdInputChunk {
            chunk: ptr,
            owned: false,
            phantom: PhantomData,
        })
    }

    /// Get total number of tokens across all chunks.
    ///
    /// This is useful for keeping track of KV cache size.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        unsafe { llama_cpp_sys_2::mtmd_helper_get_n_tokens(self.chunks.as_ptr()) }
    }

    /// Get total position count across all chunks.
    ///
    /// This is useful to keep track of `n_past`. Normally `n_pos` equals `n_tokens`,
    /// but for M-RoPE it is different.
    #[must_use]
    pub fn total_positions(&self) -> i32 {
        unsafe { llama_cpp_sys_2::mtmd_helper_get_n_pos(self.chunks.as_ptr()) }
    }

    /// Evaluate chunks using the multimodal context and LLAMA context.
    ///
    /// This helper function automatically:
    /// 1. Runs `llama_decode()` on text chunks
    /// 2. Runs `mtmd_encode()` on image chunks, then `mtmd_get_output_embd()` and then `llama_decode()`
    ///
    /// If any of the `mtmd_encode()` or `llama_decode()` calls return non-zero, the function
    /// stops and forwards the error.
    ///
    /// # Arguments
    ///
    /// * `mtmd_ctx` - The multimodal context
    /// * `llama_ctx` - The LLAMA context
    /// * `n_past` - Current position in the sequence
    /// * `seq_id` - Sequence ID for the batch
    /// * `n_batch` - Batch size for processing
    /// * `logits_last` - Whether to compute logits for the last token only
    ///
    /// # Returns
    ///
    /// Returns the new `n_past` value on success.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEvalError::EvalFailure` if any encoding or decoding operation fails.
    ///
    /// This function is NOT thread-safe.
    pub fn eval_chunks(
        &self,
        mtmd_ctx: &mut MtmdContext,
        llama_ctx: &mut LlamaContext,
        n_past: llama_cpp_sys_2::llama_pos,
        seq_id: llama_cpp_sys_2::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_sys_2::llama_pos, MtmdEvalError> {
        let mut new_n_past: llama_cpp_sys_2::llama_pos = 0;

        let result = unsafe {
            llama_cpp_sys_2::mtmd_helper_eval_chunks(
                mtmd_ctx.context.as_mut_ptr(),
                llama_ctx.context.as_mut_ptr(),
                self.chunks.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &raw mut new_n_past,
            )
        };

        if result == 0 {
            Ok(new_n_past)
        } else {
            Err(MtmdEvalError::EvalFailure(result))
        }
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::mtmd_input_chunks_free(self.chunks.as_mut_ptr()) }
    }
}

/// Safe wrapper around `mtmd_input_chunk`.
///
/// Represents a single chunk of input data, which can be either text tokens,
/// image tokens, or audio tokens. The chunk type determines what kind of
/// data and operations are available.
///
/// `'a` is the lifetime of the [`MtmdInputChunks`] this chunk was borrowed
/// from via [`MtmdInputChunks::get`]. [`MtmdInputChunk::copy`] returns an
/// owned chunk with a `'static` lifetime that is freed independently.
#[derive(Debug)]
pub struct MtmdInputChunk<'a> {
    chunk: Ptr<llama_cpp_sys_2::mtmd_input_chunk>,
    owned: bool,
    phantom: PhantomData<&'a MtmdInputChunks>,
}

// SAFETY: `mtmd_input_chunks` is thread-safe, it's POD.
unsafe impl Send for MtmdInputChunk<'_> {}
// SAFETY: Same as above.
unsafe impl Sync for MtmdInputChunk<'_> {}

impl MtmdInputChunk<'_> {
    /// Get the type of this chunk
    #[must_use]
    pub fn chunk_type(&self) -> MtmdInputChunkType {
        let chunk_type = unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_type(self.chunk.as_ptr()) };
        MtmdInputChunkType::from(chunk_type)
    }

    /// Get text tokens from this chunk.
    ///
    /// Only valid for text chunks. Returns `None` for image or audio chunks.
    ///
    /// # Returns
    ///
    /// Returns `Some(&[LlamaToken])` for text chunks, `None` otherwise.
    #[must_use]
    pub fn text_tokens(&self) -> Option<&[LlamaToken]> {
        if self.chunk_type() != MtmdInputChunkType::Text {
            return None;
        }

        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            llama_cpp_sys_2::mtmd_input_chunk_get_tokens_text(
                self.chunk.as_ptr(),
                &raw mut n_tokens,
            )
        };

        if tokens_ptr.is_null() || n_tokens == 0 {
            None
        } else {
            unsafe {
                Some(slice::from_raw_parts(
                    tokens_ptr.cast::<LlamaToken>(),
                    n_tokens,
                ))
            }
        }
    }

    /// Get the number of tokens in this chunk
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) }
    }

    /// Get the number of positions in this chunk.
    ///
    /// Returns the number of temporal positions (always 1 for M-RoPE, `n_tokens` otherwise).
    #[must_use]
    pub fn n_positions(&self) -> i32 {
        unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    /// Get chunk ID if available.
    ///
    /// Returns `None` for text chunks, may return an ID for image/audio chunks.
    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_id(self.chunk.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
                .into()
        }
    }

    /// Evaluate this single chunk (text, image, or audio) and return the updated `n_past`.
    ///
    /// Calling this chunk by chunk allows checking cancellation between chunks.
    ///
    /// # Errors
    /// Returns `MtmdEvalError::EvalFailure` if encoding or decoding fails.
    pub fn eval_single(
        &self,
        mtmd_ctx: &mut MtmdContext,
        llama_ctx: &mut LlamaContext,
        n_past: llama_cpp_sys_2::llama_pos,
        seq_id: llama_cpp_sys_2::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_sys_2::llama_pos, MtmdEvalError> {
        let mut new_n_past: llama_cpp_sys_2::llama_pos = n_past;
        let result = unsafe {
            llama_cpp_sys_2::mtmd_helper_eval_chunk_single(
                mtmd_ctx.context.as_mut_ptr(),
                llama_ctx.context.as_mut_ptr(),
                self.chunk.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &raw mut new_n_past,
            )
        };
        if result == 0 {
            Ok(new_n_past)
        } else {
            Err(MtmdEvalError::EvalFailure(result))
        }
    }

    /// Create a copy of this chunk that you own.
    ///
    /// This is useful if you want to use custom logic to handle the chunk
    /// (e.g., KV cache management) by moving the chunk ownership to your own code.
    /// Remember to ensure the copied chunk is properly freed when you're done with it.
    ///
    /// The copy does not borrow from the original collection, so it can
    /// outlive the [`MtmdInputChunks`] this chunk was taken from.
    ///
    /// # Returns
    ///
    /// Returns an owned copy of the chunk.
    ///
    /// # Errors
    ///
    /// Returns `MtmdInputChunkError::NullResult` if copying fails.
    pub fn copy(&self) -> Result<MtmdInputChunk<'static>, MtmdInputChunkError> {
        let chunk = unsafe { llama_cpp_sys_2::mtmd_input_chunk_copy(self.chunk.as_ptr()) };
        let chunk = Ptr::new(chunk).ok_or(MtmdInputChunkError::NullResult)?;
        Ok(MtmdInputChunk {
            chunk,
            owned: true,
            phantom: PhantomData,
        })
    }
}

impl Drop for MtmdInputChunk<'_> {
    fn drop(&mut self) {
        if self.owned {
            unsafe { llama_cpp_sys_2::mtmd_input_chunk_free(self.chunk.as_mut_ptr()) }
        }
    }
}

/// Outcome of attempting to add a media chunk to an [`MtmdBatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtmdBatchAddOutcome {
    /// Chunk was added to the batch.
    Added,
    /// Batch token limit was reached or the model does not support multi-chunk batching.
    BatchFull,
    /// Chunk shape or modality cannot be batched with the existing entries.
    Incompatible,
}

/// Safe wrapper around `mtmd_batch` for encoding one or more compatible media chunks together.
#[derive(Debug)]
pub struct MtmdBatch<'ctx> {
    batch: Ptr<llama_cpp_sys_2::mtmd_batch>,
    ctx: &'ctx mut MtmdContext,
}

impl<'ctx> MtmdBatch<'ctx> {
    /// Initializes an empty media batch bound to `ctx`.
    ///
    /// # Errors
    /// Returns `MtmdEncodeError::EncodeFailure` if native allocation returns null.
    pub fn new(ctx: &'ctx mut MtmdContext) -> Result<Self, MtmdEncodeError> {
        let batch = unsafe { llama_cpp_sys_2::mtmd_batch_init(ctx.context.as_mut_ptr()) };
        let batch = Ptr::new(batch).ok_or(MtmdEncodeError::EncodeFailure(1))?;
        Ok(Self { batch, ctx })
    }

    /// Adds a media chunk to this batch. The caller must keep `chunk` alive until encoding finishes.
    ///
    /// # Errors
    /// Returns `MtmdEncodeError::EncodeFailure` on invalid (e.g. text) chunks.
    pub fn add_chunk(
        &mut self,
        chunk: &MtmdInputChunk<'_>,
    ) -> Result<MtmdBatchAddOutcome, MtmdEncodeError> {
        let res = unsafe {
            llama_cpp_sys_2::mtmd_batch_add_chunk(self.batch.as_mut_ptr(), chunk.chunk.as_ptr())
        };
        match res {
            0 => Ok(MtmdBatchAddOutcome::Added),
            2 => Ok(MtmdBatchAddOutcome::BatchFull),
            3 => Ok(MtmdBatchAddOutcome::Incompatible),
            other => Err(MtmdEncodeError::EncodeFailure(other)),
        }
    }

    /// Encodes all chunks currently stored in this batch.
    ///
    /// # Errors
    /// Returns `MtmdEncodeError::EncodeFailure` if native encoding fails.
    pub fn encode(&mut self) -> Result<(), MtmdEncodeError> {
        let res = unsafe { llama_cpp_sys_2::mtmd_batch_encode(self.batch.as_mut_ptr()) };
        if res == 0 {
            Ok(())
        } else {
            Err(MtmdEncodeError::EncodeFailure(res))
        }
    }

    /// Decodes a previously batch-encoded image chunk into `llama_ctx` and returns the new `n_past`.
    ///
    /// # Errors
    /// Returns `MtmdEvalError::EvalFailure` if embeddings are missing or decoding fails.
    pub fn decode_image_chunk(
        &mut self,
        llama_ctx: &mut LlamaContext,
        chunk: &MtmdInputChunk<'_>,
        n_past: llama_cpp_sys_2::llama_pos,
        seq_id: llama_cpp_sys_2::llama_seq_id,
        n_batch: i32,
    ) -> Result<llama_cpp_sys_2::llama_pos, MtmdEvalError> {
        let embd = unsafe {
            llama_cpp_sys_2::mtmd_batch_get_output_embd(
                self.batch.as_mut_ptr(),
                chunk.chunk.as_ptr(),
            )
        };
        if embd.is_null() {
            return Err(MtmdEvalError::EvalFailure(-1));
        }
        let mut new_n_past: llama_cpp_sys_2::llama_pos = n_past;
        let res = unsafe {
            llama_cpp_sys_2::mtmd_helper_decode_image_chunk(
                self.ctx.context.as_mut_ptr(),
                llama_ctx.context.as_mut_ptr(),
                chunk.chunk.as_ptr(),
                embd,
                n_past,
                seq_id,
                n_batch,
                &raw mut new_n_past,
                None,
                std::ptr::null_mut(),
            )
        };
        if res == 0 {
            Ok(new_n_past)
        } else {
            Err(MtmdEvalError::EvalFailure(res))
        }
    }
}

impl Drop for MtmdBatch<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::mtmd_batch_free(self.batch.as_mut_ptr()) }
    }
}

/// Get the default media marker string.
///
/// Returns the default marker used to identify media positions in text
/// (typically `"<__media__>"`). This marker should be used in your input text
/// to indicate where media content should be inserted.
///
/// # Returns
///
/// Returns the default media marker as a string slice.
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::mtmd_default_marker;
///
/// let marker = mtmd_default_marker();
/// assert!(!marker.is_empty());
///
/// let text = format!("Describe this image: {}", marker);
/// assert!(text.contains(marker));
/// ```
#[must_use]
pub fn mtmd_default_marker() -> &'static str {
    unsafe {
        let c_str = llama_cpp_sys_2::mtmd_default_marker();
        CStr::from_ptr(c_str).to_str().unwrap_or("<__media__>")
    }
}

// Error types
/// Errors that can occur when initializing MTMD context
#[derive(thiserror::Error, Debug)]
pub enum MtmdInitError {
    /// Failed to create `CString` from input
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// MTMD context initialization returned null
    #[error("MTMD context initialization returned null")]
    NullResult,
}

/// Errors that can occur when working with MTMD bitmaps
#[derive(thiserror::Error, Debug)]
pub enum MtmdBitmapError {
    /// Failed to create `CString` from input
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// Invalid data size for bitmap
    #[error("Invalid data size for bitmap")]
    InvalidDataSize,
    /// Bitmap creation returned null
    #[error("Bitmap creation returned null")]
    NullResult,
}

/// Errors that can occur when working with MTMD input chunks collections
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunksError {
    /// Input chunks creation returned null
    #[error("Input chunks creation returned null")]
    NullResult,
}

/// Errors that can occur when working with individual MTMD input chunks
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunkError {
    /// Input chunk operation returned null
    #[error("Input chunk operation returned null")]
    NullResult,
}

/// Errors that can occur during tokenization
#[derive(thiserror::Error, Debug)]
pub enum MtmdTokenizeError {
    /// Number of bitmaps does not match number of markers in text
    #[error("Number of bitmaps does not match number of markers")]
    BitmapCountMismatch,
    /// Image preprocessing error occurred
    #[error("Image preprocessing error")]
    ImagePreprocessingError,
    /// Text contains characters that cannot be converted to C string
    #[error("Failed to create CString from text: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// Unknown error occurred during tokenization
    #[error("Unknown error: {0}")]
    UnknownError(i32),
}

/// Errors that can occur during encoding
#[derive(thiserror::Error, Debug)]
pub enum MtmdEncodeError {
    /// Encode operation failed
    #[error("Encode failed with code: {0}")]
    EncodeFailure(i32),
}

/// Errors that can occur during evaluation
#[derive(thiserror::Error, Debug)]
pub enum MtmdEvalError {
    /// Evaluation operation failed
    #[error("Eval failed with code: {0}")]
    EvalFailure(i32),
}

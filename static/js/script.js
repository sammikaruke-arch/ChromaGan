/**
 * ChromaGan: Interactive Frontend Controller
 * Handles drag-and-drop, sample selection, asynchronous inference requests,
 * and the interactive before/after split slider.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const imageInput = document.getElementById("imageInput");
    const previewContainer = document.getElementById("previewContainer");
    const previewImage = document.getElementById("previewImage");
    const previewMeta = document.getElementById("previewMeta");
    const clearImageBtn = document.getElementById("clearImageBtn");
    const colorizeBtn = document.getElementById("colorizeBtn");
    const actionHint = document.getElementById("actionHint");

    // View States
    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const resultState = document.getElementById("resultState");
    const loadingStatusText = document.getElementById("loadingStatusText");

    // Results & Slider Elements
    const viewToggleGroup = document.getElementById("viewToggleGroup");
    const btnSliderView = document.getElementById("btnSliderView");
    const btnSideBySideView = document.getElementById("btnSideBySideView");
    const sliderView = document.getElementById("sliderView");
    const sideBySideView = document.getElementById("sideBySideView");

    const sliderColorImg = document.getElementById("sliderColorImg");
    const sliderClip = document.getElementById("sliderClip");
    const sliderGrayImg = document.getElementById("sliderGrayImg");
    const sliderHandle = document.getElementById("sliderHandle");

    const sideBySideGray = document.getElementById("sideBySideGray");
    const sideBySideColor = document.getElementById("sideBySideColor");

    const metricResolution = document.getElementById("metricResolution");
    const metricLatency = document.getElementById("metricLatency");
    const metricDevice = document.getElementById("metricDevice");
    const metricEngine = document.getElementById("metricEngine");
    const downloadBtn = document.getElementById("downloadBtn");
    const resetBtn = document.getElementById("resetBtn");

    // State variables
    let selectedFile = null;
    let selectedSampleName = null;
    let isDraggingSlider = false;

    function getSelectedEngine() {
        return "eccv16";
    }

    // -------------------------------------------------------------------------
    // Drag and Drop Listeners
    // -------------------------------------------------------------------------
    if (dropzone) {
        ["dragenter", "dragover"].forEach(event => {
            dropzone.addEventListener(event, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(event => {
            dropzone.addEventListener(event, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                handleSelectedFile(files[0]);
            }
        });
    }

    if (imageInput) {
        imageInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleSelectedFile(e.target.files[0]);
            }
        });
    }

    if (clearImageBtn) {
        clearImageBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            resetUpload();
        });
    }

    function resetUpload() {
        selectedFile = null;
        selectedSampleName = null;
        if (imageInput) imageInput.value = "";
        if (previewImage) previewImage.src = "";
        if (previewContainer) previewContainer.classList.add("d-none");
        if (dropzone) {
            const content = dropzone.querySelector(".dropzone-content");
            if (content) content.classList.remove("d-none");
        }
        if (colorizeBtn) colorizeBtn.disabled = true;
        if (actionHint) actionHint.textContent = "Please select or drop an image above to begin";
        document.querySelectorAll(".sample-thumbnail-card").forEach(c => c.classList.remove("border-primary"));
    }

    function handleSelectedFile(file) {
        // Validate type
        const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
        if (!validTypes.includes(file.type)) {
            alert("Invalid file format. Please upload a JPG, PNG, WEBP, or BMP image.");
            return;
        }

        // Validate size (max 16 MB)
        if (file.size > 16 * 1024 * 1024) {
            alert("File is too large! Maximum allowed upload size is 16MB.");
            return;
        }

        selectedFile = file;
        selectedSampleName = null;
        document.querySelectorAll(".sample-thumbnail-card").forEach(c => c.classList.remove("border-primary"));

        // Load preview
        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImage) previewImage.src = e.target.result;
            if (dropzone) {
                const content = dropzone.querySelector(".dropzone-content");
                if (content) content.classList.add("d-none");
            }
            if (previewContainer) previewContainer.classList.remove("d-none");

            // Measure dimensions
            const img = new Image();
            img.onload = () => {
                if (previewMeta) {
                    previewMeta.textContent = `${img.width}×${img.height} px • ${(file.size / 1024).toFixed(1)} KB`;
                }
            };
            img.src = e.target.result;

            if (colorizeBtn) colorizeBtn.disabled = false;
            if (actionHint) actionHint.textContent = "Ready! Click below to synthesize colors.";
        };
        reader.readAsDataURL(file);
    }

    // -------------------------------------------------------------------------
    // Quick Demo Sample Selector
    // -------------------------------------------------------------------------
    window.selectSample = function(sampleName, sampleUrl) {
        selectedFile = null;
        selectedSampleName = sampleName;

        document.querySelectorAll(".sample-thumbnail-card").forEach(c => {
            c.classList.toggle("border-primary", c.dataset.sampleName === sampleName);
        });

        if (previewImage) previewImage.src = sampleUrl;
        if (dropzone) {
            const content = dropzone.querySelector(".dropzone-content");
            if (content) content.classList.add("d-none");
        }
        if (previewContainer) previewContainer.classList.remove("d-none");
        if (previewMeta) previewMeta.textContent = `Demo Sample: ${sampleName}`;

        if (colorizeBtn) colorizeBtn.disabled = false;
        if (actionHint) actionHint.textContent = "Sample selected! Click below to synthesize colors.";
    };

    // -------------------------------------------------------------------------
    // Colorization Request & Progress Simulation
    // -------------------------------------------------------------------------
    if (colorizeBtn) {
        colorizeBtn.addEventListener("click", async () => {
            if (!selectedFile && !selectedSampleName) return;

            // Transition to Loading State
            if (emptyState) emptyState.classList.add("d-none");
            if (resultState) {
                resultState.classList.add("d-none");
                resultState.classList.remove("d-flex");
            }
            if (loadingState) {
                loadingState.classList.remove("d-none");
                loadingState.classList.add("d-flex");
            }
            if (viewToggleGroup) viewToggleGroup.classList.add("d-none");
            colorizeBtn.disabled = true;

            let progress = 0;
            if (loadingStatusText) loadingStatusText.textContent = "Processing... 0% completed";
            const progressInterval = setInterval(() => {
                if (progress < 90) {
                    progress = Math.min(progress + 5, 90);
                    if (loadingStatusText) loadingStatusText.textContent = `Processing... ${progress}% completed`;
                }
            }, 500);

            try {
                let response;
                const engineType = getSelectedEngine();

                if (selectedFile) {
                    const formData = new FormData();
                    formData.append("image", selectedFile);
                    formData.append("model_type", engineType);
                    response = await fetch("/colorize", {
                        method: "POST",
                        body: formData,
                    });
                } else {
                    response = await fetch("/colorize-sample", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ sample_name: selectedSampleName, model_type: engineType }),
                    });
                }

                clearInterval(progressInterval);

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || "Colorization process failed.");
                }

                // Populate Results
                renderResults(data);

            } catch (error) {
                clearInterval(progressInterval);
                if (loadingState) {
                    loadingState.classList.remove("d-flex");
                    loadingState.classList.add("d-none");
                }
                if (emptyState) emptyState.classList.remove("d-none");
                colorizeBtn.disabled = false;

                alert(`Error: ${error.message}`);
            }
        });
    }

    function renderResults(data) {
        // Switch to Result View
        if (loadingState) {
            loadingState.classList.remove("d-flex");
            loadingState.classList.add("d-none");
        }
        if (resultState) {
            resultState.classList.remove("d-none");
            resultState.classList.add("d-flex");
        }
        if (viewToggleGroup) viewToggleGroup.classList.remove("d-none");
        if (colorizeBtn) colorizeBtn.disabled = false;

        // Set Images for Slider View
        if (sliderColorImg) sliderColorImg.src = data.output_url;
        if (sliderGrayImg) sliderGrayImg.src = data.input_url;

        // Set Images for Side-by-Side View
        if (sideBySideGray) sideBySideGray.src = data.input_url;
        if (sideBySideColor) sideBySideColor.src = data.output_url;

        // Set Metrics
        if (metricResolution) metricResolution.textContent = data.original_size;
        if (metricLatency) metricLatency.textContent = `${data.inference_time_ms} ms`;
        if (metricDevice) metricDevice.textContent = data.device;
        if (metricEngine) metricEngine.textContent = data.model_type || "ECCV16";

        // Set Download Link
        if (downloadBtn) downloadBtn.href = `/download/${data.download_filename}`;

        // Initialize slider at 50%
        setSliderPosition(50);
        syncSliderImagesWidth();
    }

    // -------------------------------------------------------------------------
    // Interactive Split Comparison Slider Logic
    // -------------------------------------------------------------------------
    function syncSliderImagesWidth() {
        if (!sliderView) return;
        const rect = sliderView.getBoundingClientRect();
        if (sliderGrayImg) {
            sliderGrayImg.style.width = `${rect.width}px`;
        }
    }

    window.addEventListener("resize", syncSliderImagesWidth);

    function setSliderPosition(percentage) {
        percentage = Math.max(0, Math.min(100, percentage));
        if (sliderClip) sliderClip.style.width = `${percentage}%`;
        if (sliderHandle) sliderHandle.style.left = `${percentage}%`;
    }

    function handleSliderMove(e) {
        if (!isDraggingSlider || !sliderView) return;
        const rect = sliderView.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        if (clientX === undefined) return;

        const offsetX = clientX - rect.left;
        const percentage = (offsetX / rect.width) * 100;
        setSliderPosition(percentage);
    }

    if (sliderView) {
        sliderView.addEventListener("mousedown", (e) => {
            isDraggingSlider = true;
            handleSliderMove(e);
        });

        sliderView.addEventListener("touchstart", (e) => {
            isDraggingSlider = true;
            handleSliderMove(e);
        }, { passive: true });
    }

    window.addEventListener("mousemove", handleSliderMove);
    window.addEventListener("mouseup", () => {
        isDraggingSlider = false;
    });

    window.addEventListener("touchmove", handleSliderMove, { passive: true });
    window.addEventListener("touchend", () => {
        isDraggingSlider = false;
    });

    // -------------------------------------------------------------------------
    // View Toggle Buttons (Split Slider vs Side-by-Side)
    // -------------------------------------------------------------------------
    if (btnSliderView && btnSideBySideView) {
        btnSliderView.addEventListener("click", () => {
            btnSliderView.classList.add("active");
            btnSideBySideView.classList.remove("active");
            if (sliderView) sliderView.classList.remove("d-none");
            if (sideBySideView) sideBySideView.classList.add("d-none");
            syncSliderImagesWidth();
        });

        btnSideBySideView.addEventListener("click", () => {
            btnSideBySideView.classList.add("active");
            btnSliderView.classList.remove("active");
            if (sliderView) sliderView.classList.add("d-none");
            if (sideBySideView) sideBySideView.classList.remove("d-none");
        });
    }

    // -------------------------------------------------------------------------
    // Reset Action
    // -------------------------------------------------------------------------
    if (resetBtn) {
        resetBtn.addEventListener("click", () => {
            resetUpload();
            if (resultState) {
                resultState.classList.add("d-none");
                resultState.classList.remove("d-flex");
            }
            if (emptyState) emptyState.classList.remove("d-none");
            if (viewToggleGroup) viewToggleGroup.classList.add("d-none");
        });
    }
});

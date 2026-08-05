/**
 * Medical PET Image Enhancement System - Client-Side JavaScript
 * Siemens DICOM .IMA Support & Interactive Medical Inspector
 */

document.addEventListener("DOMContentLoaded", () => {
    let selectedFile = null;
    let selectedSampleId = null;
    let histogramChartInstance = null;

    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("petFileInput");
    const enhanceBtn = document.getElementById("enhanceBtn");
    const presetBtns = document.querySelectorAll(".btn-sample-preset");

    // Preset Sample Selection Handler
    presetBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            presetBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            selectedSampleId = btn.getAttribute("data-sample-id");
            selectedFile = null;

            // Show selected badge
            const fileNameEl = document.getElementById("selectedFileName");
            const fileSizeEl = document.getElementById("selectedFileSize");
            const fileBadge = document.getElementById("fileBadge");

            if (fileNameEl) fileNameEl.textContent = `Siemens .IMA Preset (${selectedSampleId})`;
            if (fileSizeEl) fileSizeEl.textContent = "Paired Low-Dose PET Dataset Slice";
            if (fileBadge) fileBadge.classList.remove("d-none");

            if (enhanceBtn) enhanceBtn.disabled = false;
        });
    });

    // Drag and Drop Events
    if (dropzone && fileInput) {
        ["dragenter", "dragover"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelected(files[0]);
            }
        });

        dropzone.addEventListener("click", () => {
            fileInput.click();
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleFileSelected(e.target.files[0]);
            }
        });
    }

    function handleFileSelected(file) {
        selectedFile = file;
        selectedSampleId = null;
        presetBtns.forEach(b => b.classList.remove("active"));

        const fileNameEl = document.getElementById("selectedFileName");
        const fileSizeEl = document.getElementById("selectedFileSize");
        const fileBadge = document.getElementById("fileBadge");

        if (fileNameEl) fileNameEl.textContent = file.name;
        if (fileSizeEl) fileSizeEl.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";
        if (fileBadge) fileBadge.classList.remove("d-none");

        if (enhanceBtn) enhanceBtn.disabled = false;
    }

    // Enhance Action Button Click
    async function runEnhancement() {
            if (!selectedFile && !selectedSampleId) {
                alert("Please select a Siemens .IMA preset slice or upload a DICOM file first.");
                return;
            }

            const colormapSelect = document.getElementById("colormapSelect");
            const colormap = colormapSelect ? colormapSelect.value : "gray";

            const progressSection = document.getElementById("progressSection");
            const resultsSection = document.getElementById("resultsSection");
            const defaultContainer = document.getElementById("defaultContainer");
            const errorAlert = document.getElementById("errorAlert");

            if (defaultContainer) defaultContainer.classList.add("d-none");
            if (progressSection) progressSection.classList.remove("d-none");
            if (resultsSection) resultsSection.classList.add("d-none");
            if (errorAlert) errorAlert.classList.add("d-none");

            if (enhanceBtn) enhanceBtn.disabled = true;

            const formData = new FormData();
            if (selectedSampleId) {
                formData.append("sample_id", selectedSampleId);
            } else if (selectedFile) {
                formData.append("file", selectedFile);
            }
            formData.append("colormap", colormap);

            try {
                updateProgressStep(1, "Reading Siemens .IMA DICOM File & Header Tags...");
                await delay(300);

                updateProgressStep(2, "Min-Max Normalizing Voxel Intensities [0, 1]...");
                await delay(350);

                updateProgressStep(3, "Running U-Net + applying " + colormap + " colormap...");

                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 120000);

                const response = await fetch("/api/enhance", {
                    method: "POST",
                    body: formData,
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

                const rawText = await response.text();
                let data;
                try {
                    data = JSON.parse(rawText);
                } catch (_) {
                    if (!rawText || response.status === 502 || response.status === 503) {
                        throw new Error(
                            "Server crashed or timed out during enhancement (common on Render free/low RAM). " +
                            "Wait for redeploy, hard-refresh (Ctrl+F5), then try a preset sample. Status: " + response.status
                        );
                    }
                    throw new Error("Invalid server response (status " + response.status + "). Check Render logs.");
                }

                if (!response.ok || !data.success) {
                    throw new Error(data.error || "Image enhancement failed.");
                }

                updateProgressStep(4, "Postprocessing Matrix & Rendering Colormap...");
                await delay(300);

                renderResults(data);

            } catch (err) {
                console.error("Enhancement Error:", err);
                if (errorAlert) {
                    errorAlert.textContent = err.message || "An error occurred during PET enhancement.";
                    errorAlert.classList.remove("d-none");
                }
            } finally {
                if (progressSection) progressSection.classList.add("d-none");
                if (enhanceBtn) enhanceBtn.disabled = false;
            }
    }

    if (enhanceBtn) {
        enhanceBtn.addEventListener("click", () => runEnhancement());
    }

    const colormapSelectEl = document.getElementById("colormapSelect");
    if (colormapSelectEl) {
        colormapSelectEl.addEventListener("change", () => {
            // Re-run with new palette if a sample/file is already loaded
            if (selectedFile || selectedSampleId) {
                runEnhancement();
            }
        });
    }

    function updateProgressStep(stepNum, statusText) {
        const statusEl = document.getElementById("progressStatusText");
        const progressBar = document.getElementById("progressBarInner");
        if (statusEl) statusEl.textContent = statusText;
        if (progressBar) progressBar.style.width = (stepNum * 25) + "%";
    }

    function renderResults(data) {
        const resultsSection = document.getElementById("resultsSection");
        const lowImgEl = document.getElementById("resultLowImg");
        const highImgEl = document.getElementById("resultHighImg");
        const fullImgEl = document.getElementById("resultFullImg");
        const compareImgEl = document.getElementById("resultCompareImg");
        const downloadBtn = document.getElementById("downloadBtn");
        const fullDosePanel = document.getElementById("fullDosePanel");

        const psnrVal = document.getElementById("metricPsnr");
        const ssimVal = document.getElementById("metricSsim");
        const nrmseVal = document.getElementById("metricNrmse");
        const noiseVal = document.getElementById("metricNoise");

        const cacheBust = `?t=${Date.now()}`;
        if (lowImgEl) lowImgEl.src = data.low_dose_url + cacheBust;
        if (highImgEl) highImgEl.src = data.high_dose_url + cacheBust;
        if (compareImgEl && data.comparison_url) {
            compareImgEl.src = data.comparison_url + cacheBust;
        }

        if (fullImgEl) {
            if (data.full_dose_url) {
                fullImgEl.src = data.full_dose_url + cacheBust;
                if (fullDosePanel) fullDosePanel.classList.remove("d-none");
            } else {
                // No GT: reuse enhanced as right panel reference note, keep panel visible with enhanced copy
                fullImgEl.src = data.high_dose_url + cacheBust;
                fullImgEl.alt = "U-Net Enhanced (no paired GT for this upload)";
            }
        }

        if (data.dataset_stats) {
            const s = data.dataset_stats;
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            set("statLowCount", s.low_dose_images);
            set("statFullCount", s.full_dose_images);
            set("statPairedCount", s.paired_images);
            set("statTrainCount", s.training_pairs);
            set("statValCount", s.validation_pairs);
            set("statTestCount", s.testing_pairs);
        }

        if (psnrVal) psnrVal.textContent = data.metrics.psnr + " dB";
        if (ssimVal) ssimVal.textContent = data.metrics.ssim;
        if (nrmseVal) nrmseVal.textContent = data.metrics.nrmse;
        if (noiseVal) noiseVal.textContent = data.metrics.noise_reduction + "%";

        if (downloadBtn) {
            downloadBtn.href = `/download/${data.high_filename}`;
        }

        // Setup Comparison Slider
        setupComparisonSlider(data.low_dose_url + cacheBust, data.high_dose_url + cacheBust);

        // Populate DICOM Tags Inspector Modal
        populateDicomTagsModal(data.metadata);

        // Render Chart.js Intensity Histogram
        if (data.histograms) {
            renderHistogramChart(data.histograms.low, data.histograms.high);
        }

        // Patient / modality badges
        const modalityBadge = document.getElementById("resultModalityBadge");
        const patientBadge = document.getElementById("resultPatientBadge");
        const tags = (data.metadata && data.metadata.dicom_tags) || {};
        if (modalityBadge) modalityBadge.textContent = tags.modality || "PT (Positron Emission Tomography)";
        if (patientBadge) patientBadge.textContent = "Patient ID: " + (tags.patient_id || data.metadata.file_name || "Siemens_001");

        if (resultsSection) {
            resultsSection.classList.remove("d-none");
            resultsSection.scrollIntoView({ behavior: "smooth" });
        }
    }

    function setupComparisonSlider(lowUrl, highUrl) {
        const overlayImg = document.getElementById("sliderLowOverlay");
        const baseImg = document.getElementById("sliderHighBase");

        if (overlayImg) overlayImg.src = lowUrl;
        if (baseImg) baseImg.src = highUrl;

        const sliderContainer = document.getElementById("comparisonSliderContainer");
        const sliderHandle = document.getElementById("sliderHandle");
        const overlayDiv = document.getElementById("sliderOverlayDiv");

        if (!sliderContainer || !sliderHandle || !overlayDiv) return;

        let isDragging = false;

        function setSliderPos(xPos) {
            const rect = sliderContainer.getBoundingClientRect();
            let x = xPos - rect.left;
            if (x < 0) x = 0;
            if (x > rect.width) x = rect.width;
            const pct = (x / rect.width) * 100;
            sliderHandle.style.left = pct + "%";
            overlayDiv.style.width = pct + "%";
        }

        sliderHandle.addEventListener("mousedown", () => isDragging = true);
        window.addEventListener("mouseup", () => isDragging = false);

        sliderContainer.addEventListener("mousemove", (e) => {
            if (isDragging) setSliderPos(e.clientX);
        });

        // Touch support
        sliderHandle.addEventListener("touchstart", () => isDragging = true);
        window.addEventListener("touchend", () => isDragging = false);
        sliderContainer.addEventListener("touchmove", (e) => {
            if (isDragging && e.touches[0]) setSliderPos(e.touches[0].clientX);
        });
    }

    function populateDicomTagsModal(metadata) {
        const container = document.getElementById("dicomTagsContainer");
        if (!container) return;

        let html = "";
        const tags = metadata.dicom_tags || {
            "File Name": metadata.file_name,
            "Format": metadata.format,
            "Dimensions": metadata.original_shape ? metadata.original_shape.join(" x ") : "256 x 256"
        };

        for (const [key, val] of Object.entries(tags)) {
            const formattedKey = key.replace(/_/g, " ").toUpperCase();
            html += `
                <div class="col-md-6">
                    <div class="p-2.5 bg-white border rounded">
                        <small class="text-muted d-block font-monospace">${formattedKey}</small>
                        <strong class="text-primary-dark">${val}</strong>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    function renderHistogramChart(lowHist, highHist) {
        const ctx = document.getElementById("histogramChart");
        if (!ctx) return;

        if (histogramChartInstance) {
            histogramChartInstance.destroy();
        }

        histogramChartInstance = new Chart(ctx, {
            type: "line",
            data: {
                labels: lowHist.labels,
                datasets: [
                    {
                        label: "Low-Dose Photon Noise",
                        data: lowHist.values,
                        borderColor: "#F59E0B",
                        backgroundColor: "rgba(245, 158, 11, 0.1)",
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: "U-Net High-Dose Restored Signal",
                        data: highHist.values,
                        borderColor: "#00A896",
                        backgroundColor: "rgba(0, 168, 150, 0.15)",
                        fill: true,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "top" }
                },
                scales: {
                    x: { title: { display: true, text: "Normalized SUV Voxel Intensity [0 - 1]" } },
                    y: { title: { display: true, text: "Voxel Count" } }
                }
            }
        });
    }

    // 3D Series Navigator Range Slider Handler
    const sliceRange = document.getElementById("sliceRange");
    const sliceDisplayNum = document.getElementById("sliceDisplayNum");
    const prevSliceBtn = document.getElementById("prevSliceBtn");
    const nextSliceBtn = document.getElementById("nextSliceBtn");

    if (sliceRange && sliceDisplayNum) {
        sliceRange.addEventListener("input", (e) => {
            const val = e.target.value;
            sliceDisplayNum.textContent = `Slice #${val} / 50`;
        });

        if (prevSliceBtn) {
            prevSliceBtn.addEventListener("click", () => {
                if (parseInt(sliceRange.value) > 1) {
                    sliceRange.value = parseInt(sliceRange.value) - 1;
                    sliceDisplayNum.textContent = `Slice #${sliceRange.value} / 50`;
                }
            });
        }

        if (nextSliceBtn) {
            nextSliceBtn.addEventListener("click", () => {
                if (parseInt(sliceRange.value) < 50) {
                    sliceRange.value = parseInt(sliceRange.value) + 1;
                    sliceDisplayNum.textContent = `Slice #${sliceRange.value} / 50`;
                }
            });
        }
    }

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
});

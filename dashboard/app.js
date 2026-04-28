/* ======================================================================
   ChestAI Clinical Dashboard — Application Logic
   ====================================================================== */

(function () {
    "use strict";

    // --- DOM refs --------------------------------------------------------
    const $ = (sel) => document.querySelector(sel);
    const uploadArea     = $("#upload-area");
    const uploadContent  = $("#upload-content");
    const uploadPreview  = $("#upload-preview");
    const previewImage   = $("#preview-image");
    const fileInput      = $("#file-input");
    const btnClear       = $("#btn-clear");
    const btnAnalyze     = $("#btn-analyze");
    const patientIdInput = $("#patient-id");
    const headerStatus   = $("#header-status");

    const uploadSection  = $("#upload-section");
    const loadingSection = $("#loading-section");
    const resultsSection = $("#results-section");

    const overlaySlider  = $("#overlay-slider");
    const overlayValue   = $("#overlay-value");
    const btnCopyReport  = $("#btn-copy-report");
    const btnNewAnalysis = $("#btn-new-analysis");

    let selectedFile = null;

    // --- Health check ----------------------------------------------------
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await res.json();
                headerStatus.className = "header-status online";
                headerStatus.innerHTML = `<span class="status-dot"></span><span>Model: ${data.model_name}</span>`;
            } else {
                throw new Error("not ok");
            }
        } catch {
            headerStatus.className = "header-status error";
            headerStatus.innerHTML = '<span class="status-dot"></span><span>Model unavailable</span>';
        }
    }
    checkHealth();

    // --- File handling ----------------------------------------------------
    function handleFile(file) {
        if (!file || !file.type.startsWith("image/")) return;
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = "none";
            uploadPreview.style.display = "flex";
            btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearFile() {
        selectedFile = null;
        previewImage.src = "";
        uploadContent.style.display = "";
        uploadPreview.style.display = "none";
        btnAnalyze.disabled = true;
        fileInput.value = "";
    }

    // Click to upload
    uploadArea.addEventListener("click", (e) => {
        if (e.target === btnClear || e.target.closest(".btn-clear")) return;
        if (uploadPreview.style.display !== "none") return;
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    btnClear.addEventListener("click", (e) => {
        e.stopPropagation();
        clearFile();
    });

    // Drag & drop
    ["dragenter", "dragover"].forEach((evt) =>
        uploadArea.addEventListener(evt, (e) => {
            e.preventDefault();
            uploadArea.classList.add("drag-over");
        })
    );
    ["dragleave", "drop"].forEach((evt) =>
        uploadArea.addEventListener(evt, () => uploadArea.classList.remove("drag-over"))
    );
    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    // --- Analyze ---------------------------------------------------------
    btnAnalyze.addEventListener("click", runAnalysis);

    async function runAnalysis() {
        if (!selectedFile) return;

        // Show loading
        uploadSection.style.display = "none";
        resultsSection.style.display = "none";
        loadingSection.style.display = "";

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("patient_id", patientIdInput.value || "ANONYMOUS");

        try {
            const res = await fetch("/analyze", { method: "POST", body: formData });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: "Unknown error" }));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            renderResults(data);
        } catch (err) {
            alert("Analysis failed: " + err.message);
            loadingSection.style.display = "none";
            uploadSection.style.display = "";
        }
    }

    // --- Render results --------------------------------------------------
    function renderResults(data) {
        loadingSection.style.display = "none";
        resultsSection.style.display = "";

        const sev = data.severity;
        const unc = data.uncertainty;
        const reg = data.regions;
        const met = data.clinical_metrics;

        // Severity banner
        const banner = $("#severity-banner");
        banner.className = `severity-banner severity-${sev.level} animate-in`;
        $("#severity-badge").textContent = sev.label;
        $("#result-prediction").textContent = data.prediction;
        $("#severity-description").textContent = sev.description;
        $("#severity-score").textContent = `${sev.level} / 5`;

        // Gauge dots
        const dots = document.querySelectorAll(".gauge-dot");
        const sevColors = ["", "var(--severity-1)", "var(--severity-2)", "var(--severity-3)", "var(--severity-4)", "var(--severity-5)"];
        dots.forEach((dot) => {
            const level = parseInt(dot.dataset.level);
            if (level <= sev.level) {
                dot.classList.add("active");
                dot.style.background = sevColors[level];
                dot.style.borderColor = "transparent";
                dot.style.boxShadow = `0 0 8px ${sevColors[level]}`;
            } else {
                dot.classList.remove("active");
                dot.style.background = "";
                dot.style.borderColor = "";
                dot.style.boxShadow = "";
            }
        });

        // Metrics cards
        const pProb = data.pneumonia_probability;
        $("#metric-pneumonia-prob").textContent = (pProb * 100).toFixed(1) + "%";
        $("#bar-pneumonia").style.width = (pProb * 100) + "%";

        const stdPct = (unc.std_probability * 100).toFixed(1);
        $("#metric-uncertainty").textContent = `±${stdPct}%`;
        if (unc.std_probability < 0.05) {
            $("#uncertainty-desc").textContent = "Low — model is confident";
            $("#uncertainty-desc").style.color = "var(--green)";
        } else if (unc.std_probability < 0.10) {
            $("#uncertainty-desc").textContent = "Moderate — consider correlation";
            $("#uncertainty-desc").style.color = "var(--amber)";
        } else {
            $("#uncertainty-desc").textContent = "High — senior review recommended";
            $("#uncertainty-desc").style.color = "var(--red)";
        }

        if (met.sensitivity > 0) {
            $("#metric-sens-spec").textContent =
                `${(met.sensitivity * 100).toFixed(1)}% / ${(met.specificity * 100).toFixed(1)}%`;
            $("#metric-ppv-npv").textContent =
                `PPV ${(met.ppv * 100).toFixed(1)}%  ·  NPV ${(met.npv * 100).toFixed(1)}%`;
        } else {
            $("#metric-sens-spec").textContent = "—";
            $("#metric-ppv-npv").textContent = "Run calibrate.py for metrics";
        }

        // Image viewer
        // Use the uploaded file as the original
        const reader = new FileReader();
        reader.onload = (e) => {
            $("#viewer-original").src = e.target.result;
            $("#viewer-base").src = e.target.result;
        };
        reader.readAsDataURL(selectedFile);

        $("#viewer-heatmap").src = "data:image/png;base64," + data.gradcam_overlay_b64;

        // Reset slider
        overlaySlider.value = 50;
        overlayValue.textContent = "50%";
        updateOverlayOpacity(50);

        // Region analysis
        $("#region-primary").textContent = reg.primary_region;
        $("#region-laterality").textContent = reg.laterality;
        $("#region-area").textContent = `~${reg.affected_area_pct}%`;
        $("#region-pattern").textContent = reg.pattern;

        // Zone bars
        const zoneBarsEl = $("#zone-bars");
        zoneBarsEl.innerHTML = "";
        const zones = Object.entries(reg.zone_scores).sort((a, b) => b[1] - a[1]);
        zones.forEach(([name, score]) => {
            const pct = (score * 100).toFixed(1);
            const row = document.createElement("div");
            row.className = "zone-bar-row";
            row.innerHTML = `
                <span class="zone-bar-label">${name}</span>
                <div class="zone-bar-track">
                    <div class="zone-bar-fill" style="width: 0%"></div>
                </div>
                <span class="zone-bar-value">${pct}%</span>
            `;
            zoneBarsEl.appendChild(row);
            // Animate
            requestAnimationFrame(() => {
                row.querySelector(".zone-bar-fill").style.width = pct + "%";
            });
        });

        // Report
        $("#report-text").textContent = data.report || "No report generated.";

        // Animate sections
        document.querySelectorAll(".metrics-row .metric-card").forEach((el, i) => {
            el.classList.remove("animate-in");
            void el.offsetWidth;
            el.classList.add("animate-in");
            el.style.animationDelay = `${(i + 1) * 0.06}s`;
        });

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // --- Overlay slider --------------------------------------------------
    function updateOverlayOpacity(val) {
        const heatmap = $("#viewer-heatmap");
        if (heatmap) heatmap.style.opacity = val / 100;
    }

    overlaySlider.addEventListener("input", () => {
        const val = overlaySlider.value;
        overlayValue.textContent = val + "%";
        updateOverlayOpacity(val);
    });

    // --- Copy report -----------------------------------------------------
    btnCopyReport.addEventListener("click", () => {
        const text = $("#report-text").textContent;
        navigator.clipboard.writeText(text).then(() => {
            btnCopyReport.classList.add("copied");
            btnCopyReport.querySelector("svg + *") || null;
            const label = btnCopyReport.childNodes[btnCopyReport.childNodes.length - 1];
            const original = label.textContent;
            label.textContent = " Copied!";
            setTimeout(() => {
                btnCopyReport.classList.remove("copied");
                label.textContent = original;
            }, 2000);
        });
    });

    // --- New analysis ----------------------------------------------------
    btnNewAnalysis.addEventListener("click", () => {
        resultsSection.style.display = "none";
        uploadSection.style.display = "";
        clearFile();
        window.scrollTo({ top: 0, behavior: "smooth" });
    });
})();

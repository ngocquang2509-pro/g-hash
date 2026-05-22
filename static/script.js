document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewImage = document.getElementById('preview-image');
    
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsSection = document.getElementById('results-section');
    const predictionsContainer = document.getElementById('predictions-container');
    const retrievalGrid = document.getElementById('retrieval-grid');

    // Drag & Drop Handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', handleDrop, false);
    browseBtn.addEventListener('click', () => fileInput.click());
    
    // Bấm vào ảnh preview để chọn ảnh khác
    previewImage.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length) handleFiles(this.files);
    });

    // Lightbox Modal Logic
    const lightboxModal = document.getElementById('lightbox-modal');
    const lightboxImg = document.getElementById('lightbox-img');
    const closeLightbox = document.querySelector('.close-lightbox');

    closeLightbox.addEventListener('click', () => {
        lightboxModal.style.display = 'none';
    });
    lightboxModal.addEventListener('click', (e) => {
        if(e.target === lightboxModal) lightboxModal.style.display = 'none';
    });

    window.openLightbox = function(imgSrc) {
        lightboxImg.src = imgSrc;
        lightboxModal.style.display = 'flex';
    };

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        const file = files[0];
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Show Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            dropArea.querySelector('.upload-content').style.opacity = '0'; // Hide text but keep clickable
        }
        reader.readAsDataURL(file);

        // Upload and Analyze
        analyzeImage(file);
    }

    async function analyzeImage(file) {
        // UI State
        resultsSection.style.display = 'none';
        loadingSpinner.style.display = 'block';
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                renderPredictions(data.predictions);
                renderRetrieval(data.retrieval);
                
                loadingSpinner.style.display = 'none';
                resultsSection.style.display = 'flex'; // Trigger animation
                
                // Trigger CSS transitions for bars
                setTimeout(() => {
                    document.querySelectorAll('.pred-bar-fill').forEach(bar => {
                        bar.style.width = bar.getAttribute('data-width');
                    });
                }, 50);
            } else {
                alert('Analysis failed: ' + (data.error || 'Unknown error'));
                loadingSpinner.style.display = 'none';
            }
        } catch (error) {
            console.error('Error:', error);
            alert('A network error occurred.');
            loadingSpinner.style.display = 'none';
        }
    }

    function renderPredictions(preds) {
        predictionsContainer.innerHTML = '';
        preds.forEach(p => {
            const pct = (p.confidence * 100).toFixed(1);
            const html = `
                <div class="prediction-item">
                    <div class="pred-header">
                        <span class="pred-label">${p.label.replace(/_/g, ' ')}</span>
                        <span class="pred-pct">${pct}%</span>
                    </div>
                    <div class="pred-bar-bg">
                        <div class="pred-bar-fill" data-width="${pct}%" style="width: 0%;"></div>
                    </div>
                </div>
            `;
            predictionsContainer.insertAdjacentHTML('beforeend', html);
        });
    }

    function renderRetrieval(retrieved) {
        retrievalGrid.innerHTML = '';
        if (!retrieved || retrieved.length === 0) {
            retrievalGrid.innerHTML = '<p style="color:var(--text-secondary); grid-column:1/-1;">No similar images found in database or database not indexed.</p>';
            return;
        }

        retrieved.forEach(item => {
            const score = (item.rerank_score * 100).toFixed(1);
            const html = `
                <div class="result-card">
                    <span class="badge-rank">#${item.rank}</span>
                    <div class="card-img-wrapper" onclick="openLightbox('${item.image_url}')" title="Bấm để xem ảnh lớn">
                        <img class="card-img" src="${item.image_url}" alt="Rank ${item.rank}">
                    </div>
                    <div class="card-info">
                        <div class="info-row">
                            <span>Hamming Dist:</span>
                            <span class="info-val">${item.hamming_distance} bits</span>
                        </div>
                        <div class="info-row">
                            <span>Match Score:</span>
                            <span class="info-val">${score}%</span>
                        </div>
                    </div>
                </div>
            `;
            retrievalGrid.insertAdjacentHTML('beforeend', html);
        });
    }
});

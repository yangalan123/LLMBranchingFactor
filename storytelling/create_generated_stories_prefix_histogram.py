import os
import csv
import json
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from pathlib import Path
from wordcloud import WordCloud


def process_csv_files(directory='generated_stories/'):
    # Dictionary to store word counts and text for each model
    model_word_counts = defaultdict(list)
    model_text = defaultdict(str)
    model_vocab = defaultdict(set)

    # Download required NLTK data
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Find all CSV files matching the pattern
    csv_files = Path(directory).glob('*_response_*top_p_0.9*.csv')

    for csv_path in csv_files:
        # Extract model name from filename
        model_name = csv_path.name.split('_')[0]

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Process each response
            for row in reader:
                if 'response' in row:
                    # Tokenize and count words
                    words = word_tokenize(row['response'].lower())
                    word_count = len(words)
                    model_word_counts[model_name].append(word_count)

                    # Accumulate text for word cloud
                    model_text[model_name] += row['response'] + " "

                    # Add to vocabulary (excluding stopwords and non-alphabetic tokens)
                    model_vocab[model_name].update(
                        word for word in words
                        if word.isalpha() and word not in stop_words
                    )

    # Calculate histograms for each model
    histogram_data = {}
    wordcloud_data = {}
    for model, counts in model_word_counts.items():
        # Calculate histogram
        hist, bin_edges = np.histogram(counts, bins=30)

        # Generate word cloud
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(model_text[model])

        # Convert word cloud to base64 for embedding in HTML
        import io
        import base64
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img_str = base64.b64encode(img.getvalue()).decode()

        # Store data
        histogram_data[model] = {
            'counts': hist.tolist(),
            'bins': [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
        }
        wordcloud_data[model] = img_str

    # Calculate vocabulary sizes
    vocab_sizes = {model: len(vocab) for model, vocab in model_vocab.items()}

    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Response Analysis</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.24.2/plotly.min.js"></script>
        <style>
            .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
            .plot {{ margin-bottom: 40px; }}
            .wordcloud-grid {{ 
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            .wordcloud-item {{ 
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 5px;
            }}
            h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Word Count Distribution</h2>
            <div id="word-count-plot" class="plot"></div>

            <h2>Unique Vocabulary Usage</h2>
            <div id="vocab-plot" class="plot"></div>

            <h2>Word Clouds by Model</h2>
            <div class="wordcloud-grid">
                {"".join(f'''
                    <div class="wordcloud-item">
                        <h3>{model}</h3>
                        <img src="data:image/png;base64,{img_str}" alt="{model} wordcloud" style="width:100%">
                    </div>
                ''' for model, img_str in wordcloud_data.items())}
            </div>
        </div>

        <script>
            // Word count distribution plot
            const histData = {json.dumps(histogram_data)};
            const histTraces = [];
            for (const model in histData) {{
                histTraces.push({{
                    name: model,
                    x: histData[model].bins,
                    y: histData[model].counts,
                    type: 'bar',
                    opacity: 0.7
                }});
            }}

            const histLayout = {{
                title: 'Word Count Distribution by Model',
                xaxis: {{
                    title: 'Word Count',
                    tickangle: -45
                }},
                yaxis: {{
                    title: 'Frequency'
                }},
                barmode: 'overlay',
                showlegend: true,
                legend: {{
                    bgcolor: 'rgba(255, 255, 255, 0.8)'
                }}
            }};

            Plotly.newPlot('word-count-plot', histTraces, histLayout);

            // Vocabulary size plot
            const vocabData = {json.dumps(vocab_sizes)};
            const vocabTrace = {{
                x: Object.keys(vocabData),
                y: Object.values(vocabData),
                type: 'bar',
                marker: {{
                    color: 'rgb(158,202,225)',
                    opacity: 0.8
                }}
            }};

            const vocabLayout = {{
                title: 'Unique Vocabulary Size by Model',
                xaxis: {{
                    title: 'Model',
                    tickangle: -45
                }},
                yaxis: {{
                    title: 'Number of Unique Words'
                }},
                showlegend: false
            }};

            Plotly.newPlot('vocab-plot', [vocabTrace], vocabLayout);
        </script>
    </body>
    </html>
    """

    # Write the final HTML file
    with open('model_analysis_visualization.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("Visualization has been generated as 'model_analysis_visualization.html'")


if __name__ == "__main__":
    process_csv_files()
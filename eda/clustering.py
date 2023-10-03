from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

import plotly.express as px
import umap.umap_ as umap
import torch
import hdbscan


def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components,
                      min_cluster_size,
                      random_state=None):
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=n_components,
                                 metric='cosine',
                                 random_state=random_state)
                       .fit_transform(message_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               metric='euclidean',
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters


def score_clusters(clusters, prob_threshold=0.05):

    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num)

    return label_count, cost


from tqdm import tqdm


def random_search(embeddings, space, num_evals):

    results = []

    for i in tqdm(range(num_evals)):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])

        clusters = generate_clusters(embeddings,
                                     n_neighbors=n_neighbors,
                                     n_components=n_components,
                                     min_cluster_size=min_cluster_size,
                                     random_state=42)

        label_count, cost = score_clusters(clusters, prob_threshold=0.05)

        results.append([i, n_neighbors, n_components, min_cluster_size,
                        label_count, cost])

    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components',
                                               'min_cluster_size', 'label_count', 'cost'])

    return result_df.sort_values(by='cost')


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def get_bert_emb(list_of_texts):
    embeddings = []
    for i in range(len(list_of_texts)):
        embeddings.append(embed_bert_cls(
            list_of_texts[i], sentence_model, sentence_tokenizer))
    return np.array(embeddings)


def plot_clusters(embeddings, clusters, obj_labels=None, n_neighbors=15, min_dist=0.1):
    umap_data = umap.UMAP(n_neighbors=n_neighbors,
                          n_components=2,
                          min_dist=min_dist,
                          # metric='cosine',
                          random_state=random_state).fit_transform(embeddings)

    point_size = 100.0 / np.sqrt(embeddings.shape[0])

    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['cluster'] = clusters.labels_
    result['text'] = obj_labels
    result['text'] = result.text.apply(lambda x: preprocessDataset(x))

    # fig, ax = plt.subplots(figsize=(14, 8))
    # outliers = result[result.labels == -1]
    # clustered = result[result.labels != -1]
    # plt.scatter(outliers.x, outliers.y, color = 'lightgrey', s=point_size)
    # plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=point_size, cmap='jet')
    # plt.colorbar()
    # plt.show()

    fig = px.scatter(
        result, x='x', y='y',
        color='cluster', labels={'cluster': 'cluster'}, width=1300, height=700
    )

    for el in result.groupby(by='cluster'):
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(3, 3))
        num = vectorizer.fit_transform(el[1].text.tolist())
        id_max = num.toarray().sum(axis=0).argmax()

        # id_1 = np.argsort(num.toarray().sum(axis=0))[-1]
        # id_2 = np.argsort(num.toarray().sum(axis=0))[-2]

        cluster_text = vectorizer.get_feature_names_out()[id_max]

        fig.add_annotation(
            x=el[1].x.mean(),
            y=el[1].y.mean(),
            text=str(el[0]) + ' ' + cluster_text,
            hovertext=cluster_text,
            opacity=0.8,
            bgcolor='#FFFFFF',
            showarrow=False,
            font={'size': 15, 'color': '#000000'})

    # fig.update_traces(textfont_size=1)
    fig.show()

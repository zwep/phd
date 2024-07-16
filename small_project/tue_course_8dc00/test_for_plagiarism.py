
"""
Check if we can read in .pdfs

We can...

We were also able to detect some fraudelous texts...
Which was better when I used self-defined pieces of text instead of automatic it...

"""
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import helper.plot_class as hplotc

# ddata = '/home/bugger/Documents/TUE/AssistentDocent/Assignment1_2021/Registratrion_2021'
# ddata_2022 = '/home/bugger/Documents/TUE/AssistentDocent/Assignment1/submissions'

ddata = '/home/bugger/Documents/TUE/AssistentDocent/Assignment2_2021'
ddata_2022 = '/home/bugger/Documents/TUE/AssistentDocent/Assignment2/submissions'


def get_pdf_files(ddata):
    pdf_files = []
    for d, _, f in os.walk(ddata):
        filter_f = [x for x in f if x.endswith('pdf')]
        if len(filter_f):
            full_file = os.path.join(d, filter_f[0])
            pdf_files.append(full_file)
    return pdf_files


def prep_text(text, sentence_length=50):
    text = re.sub('\n', '', text)
    text = re.sub('\s+', ' ', text)
    page_text = text.split('.')
    page_text = [x for x in page_text if len(x) > sentence_length]
    page_text = [x for x in page_text if not x.strip().startswith('Experiment')]
    page_text = [x for x in page_text if not x.strip().startswith('Figure')]
    return page_text

project_file_list_2021 = get_pdf_files(ddata)
project_file_list_2022 = get_pdf_files(ddata_2022)

for reference_file in project_file_list_2022:
    reader = PdfReader(reference_file)
    number_of_pages = len(reader.pages)
    reference_text = []
    # Skip title page and figures etc..
    # Use even more limiting pages because of clutter...
    for i_page in range(1, number_of_pages-1):
        page = reader.pages[i_page]
        text = page.extract_text()
        page_text = prep_text(text, sentence_length=20)
        reference_text.extend(page_text)

    # # Selecting pieces of text manually because that wokred better
    # reference_file = 'group 8'
    # reference_text = ['The PreactResNet50 encoder, the HoVer branch decoder, the NP branch']
    reference_text = "Graham et. al use several components in their research that benefit the performance of the model. To gain robustness against nuclei that vary in size, they let the network process the input at multiple resolutions. Next to this, they used batch normalization. This technique normalizes activations in intermediate layers of the network, resulting in less overfitting of the network. The cells in the pathology slices may be overlapping each other. This can result in overlapping nuclei and wrong segmentations during the first network. The outliers that come from this, are accounted for by a simple measurement technique. It makes sure that the sizes of the nuclei do not become larger than a certain number, if this is the case, the nuclei are marked as overlapping. The use of multiple dense layers increases the receptive field while maintaining a minimum set of parameters. This makes the model more computationally efficient. Another way they reduced the computational time, is by the shared encoder. By sharing the encoder the information needed for the three different decoders is the same. This not only reduces the training time but also increases performance."
    reference_text = reference_text.split('. ')
    vect = TfidfVectorizer(min_df=1, stop_words="english")

    for sel_file in project_file_list_2021:
        reader = None
        try:
            reader = PdfReader(sel_file)
        except:
            print("ehm")
        if reader is None:
            print('Skipping ', sel_file)
            continue

        number_of_pages = len(reader.pages)

        for i_page in range(number_of_pages):
            page = reader.pages[i_page]
            text = page.extract_text()
            page_text = prep_text(text)
            text_to_analyze = list(np.copy(reference_text))
            n_orig = len(text_to_analyze)
            text_to_analyze.extend(page_text)
            tfidf = vect.fit_transform(text_to_analyze)
            pairwise_similarity = tfidf * tfidf.T
            pairwise_similarity[np.diag_indices_from(pairwise_similarity)] = 0
            triu_pairwise_similarity = np.triu(pairwise_similarity.toarray())
            # Klopt dit nu...?
            high_indices = np.argwhere(triu_pairwise_similarity[:n_orig, n_orig:] > 0.5)
            if len(high_indices):
                print(os.path.basename(reference_file), '//', os.path.basename(sel_file), i_page, "Number of matches", len(high_indices))
                for iind in high_indices:
                    i_orig, i_new = iind
                    print('\t\t **', text_to_analyze[i_orig],'\n\t\t oo', text_to_analyze[i_new])

# Course Search System

This project implements a **course search system** that allows users to find relevant courses based on a given query using **TF-IDF** and **Word2Vec** embeddings. The system scrapes course data from a website and allows users to search for courses interactively via a **Streamlit** application.

## Project Overview

- **Web Scraping**: The course data is scraped from the Analytics Vidhya platform using **BeautifulSoup** and **requests**.
- **Search System**: The system combines **TF-IDF** and **Word2Vec** embeddings to find courses most relevant to a search query based on cosine similarity.
- **Streamlit App**: The search system is exposed through a **Streamlit** web app that displays the course details, including title, image, and URL.

## Features

- **Search for courses**: Enter any query related to courses (e.g., "Data Science", "Machine Learning", etc.), and get the most relevant courses from the scraped dataset.
- **Display Results**: Each search result shows the course title, image, and a link to the course URL.
- **Combining Models**: The system uses a combination of **TF-IDF** and **Word2Vec** embeddings to rank courses based on relevance to the search query.

## Prerequisites

Before running the app, make sure to install the following Python packages:

- `streamlit`: For creating the web interface.
- `gensim`: For Word2Vec embedding model.
- `scikit-learn`: For TF-IDF vectorization and cosine similarity.
- `requests`: For making HTTP requests.
- `beautifulsoup4`: For parsing HTML and scraping data.
- `pandas`: For data handling.

## File Structure

- `app.py`: The Streamlit app that serves as the user interface for searching courses.
- `course_scraping.py`: Script to scrape course data from the Analytics Vidhya website.
- `requirements.txt`: List of required dependencies.

## Scraping Course Data

1. **Web Scraping**: The course titles, images, and URLs are scraped from the course catalog on the Analytics Vidhya platform. The data is stored in a structured format like a CSV or DataFrame for easy manipulation.

2. **Data Fields**: Each scraped course contains the following fields:
   - **Title**: The name of the course.
   - **Image URL**: The URL of the course's thumbnail image.
   - **Course URL**: The direct link to the course's webpage.

3. **Data Storage**: The scraped data is saved in a CSV file or a pandas DataFrame, which is later used to build the search system.

## Building the Course Search System

1. **Text Preprocessing**: The course titles are preprocessed by converting to lowercase and removing any unnecessary whitespace to standardize the text data.

2. **TF-IDF Vectorization**: The TF-IDF model is used to convert course titles into numerical vectors, allowing for cosine similarity computation between the user's query and course titles.

3. **Word2Vec Embeddings**: The **Word2Vec** model is trained on the course titles to create dense vector representations of words, which are then averaged to generate embeddings for each course title.

4. **Cosine Similarity**: The system calculates cosine similarity scores for both TF-IDF and Word2Vec embeddings to rank courses based on their relevance to the input query. The combined similarity score is used to retrieve the most relevant courses.

## Running the Streamlit App

1. **User Interface**: The Streamlit app presents a simple input field for entering search queries. Once the user enters a query, the app displays the top courses most relevant to the search.

2. **Result Display**: For each result, the course title, image, and URL are shown. The courses are ranked based on the similarity scores computed from the search query.

3. **How to Run**: 
   
   - To start the Streamlit app, use the command: 
     ```bash
     streamlit run searching.py
     ```

4. **Search Queries**: You can search for any course by entering keywords like "Data Science", "Deep Learning", "Machine Learning", or other related terms. The app will return the most relevant courses from the scraped dataset.

## Future Enhancements

- **Improved Embeddings**: Explore using pre-trained embeddings like **GloVe** or **fastText** to enhance the semantic understanding of course titles.
- **Course Metadata**: Extend the system to include additional course information such as course duration, ratings, instructors, or prerequisites.
- **Search Filtering**: Implement filters to narrow down the search results by course type, level, or language.


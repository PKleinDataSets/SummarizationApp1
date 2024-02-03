import streamlit as st
import os
import PyPDF2
from transformers import pipeline

# Load the Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_pdf(input_file, summary_percentage=0.2):
    # Read the PDF file
    with open(input_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        # Initialize a string to store the text from the PDF
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        # Calculate the target length of the summary
        target_length = int(summary_percentage * len(text))
        max_length = min(target_length, 1024)  # Set a maximum length

        # Summarize the text
        summary = summarizer(text, max_length=max_length, min_length=int(summary_percentage*len(text)*0.8), do_sample=False)[0]['summary_text']

        return summary

def main():
    st.title("PDF Summarizer")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

    if uploaded_file is not None:
        # Display uploaded file
        st.write("Uploaded PDF file:", uploaded_file.name)

        # Summarize PDF when button is clicked
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                # Get summary
                summary = summarize_pdf(uploaded_file, summary_percentage=0.2)  # You can adjust the summary percentage here

            # Show summary
            st.subheader("Summary")
            st.write(summary)

            # Download summary as text file
            with st.expander("Download Summary"):
                with open("summary.txt", "w") as file:
                    file.write(summary)
                st.write("Download your summary:")
                st.download_button(
                    label="Download Summary",
                    data=open("summary.txt", "rb").read(),
                    file_name="summary.txt",
                    mime="text/plain"
                )
                os.remove("summary.txt")  # Remove the temporary summary file

if __name__ == "__main__":
    main()


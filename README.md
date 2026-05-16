# research-paper-summarizer
A LangChain-based tool that analyzes research papers through a 3-step sequential chain: extraction → simplification → recommendation.

# 📚 Research Paper Analyzer - 3-Step Sequential Chain

A powerful LangChain-based application that automatically analyzes research papers through a 3-step sequential pipeline: extraction → simplification → recommendation. Perfect for quickly understanding if a paper is worth your time.

## 🎯 What It Does

This tool takes any research paper PDF and runs it through three intelligent chains:

1. **Extractor Chain** 📄
   - Extracts structured information (Abstract, Methodology, Findings)
   - Uses LLM as a structured data extractor

2. **Simplifier Chain** 🔤  
   - Rewrites technical content in plain English
   - Explains real-world problems and methodology importance
   - Perfect for non-experts or quick understanding

3. **Recommender Chain** ⭐
   - Provides relevance score (1-10)
   - Identifies target audience
   - Gives actionable verdict on whether to read

## 🚀 Features

- ✅ **Sequential Processing** - Each chain builds on the previous one
- ✅ **Clean Pipeline Architecture** - Using LangChain's RunnableSequence
- ✅ **Multiple LLM Support** - Works with OpenAI, Groq, Local models
- ✅ **PDF Text Extraction** - Handles various PDF formats
- ✅ **Structured Output** - Returns organized recommendations
- ✅ **Save Results** - Option to export analysis to file

## 📋 Prerequisites

- Python 3.9+
- API key for your chosen LLM provider (OpenAI/Groq)
- PDF file to analyze

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/research-paper-analyzer.git
cd research-paper-analyzer

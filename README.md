<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/tttonyalpha/news_monitoring">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">News monitoring service</h3>

  <p align="center">
    Prototype for news monitoring service 
    <br />
    <a href="https://github.com/tttonyalpha/news_monitoring"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/tttonyalpha/news_monitoring">View Demo</a>
    ·
    <a href="https://github.com/tttonyalpha/news_monitoring/issues">Report Bug</a>
    ·
    <a href="https://github.com/tttonyalpha/news_monitoring/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<!-- <details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
 -->


<!-- ABOUT THE PROJECT -->
## About The Project

The main task of news monitoring is to process the incoming stream of news, identifying events that are interesting to users. In the banking sector, this can be useful for predicting defaults of major borrowers, such as various large companies. In this case, it is necessary to build a model to detect in the news an event corresponding to a delay in putting a certain object into operation. Later, with the selected texts, simpler models can be used to search for mentions of the bank's borrowers.

## Data collection and preparation

For this project, a pre-selected set of training and testing data was used. More details about the data analysis can be found below.

The training dataset consists of 1.6k samples, with 19 percent being target texts. The remaining texts were chosen in a way that it is initially difficult to determine whether they are target texts or not.
![image](https://github.com/tttonyalpha/news_monitoring/assets/79598074/0b62eb12-63cb-4183-b0d6-52781788ef26)


The testing dataset is a set of 10k samples collected from various news sources over the course of one week.
![image](https://github.com/tttonyalpha/news_monitoring/assets/79598074/d17b66a0-fdbb-40ba-b52d-df24e4352305)


## Topic modeling

To get sentence embeddings I used model [cointegrated/rubert-tiny2](https://habr.com/ru/articles/669674/), which were trained to produce high-quality sentence embeddings. Then I reduced the dimensions of the embeddings using UMAP and clustered them using HDBSCAN. To tune hyperparameters and score clusters I used [Bayesian optimization with Hyperopt](https://github.com/hyperopt/hyperopt)

## Summarization 

For texts with more than 512 tokens, we will summarize them to fit into the classifier

#### Choosing a method: extractive vs abstractive summarization
I decided to use the abstractive model, despite the fact that extractive models work faster in this case. My choice is justified by the fact that the texts in the test set are quite large, there may be several different topics, and an extractive model may not extract what we need from such text.

#### Model selection 

I chose the model [mbart_ru_sum_gazeta](https://huggingface.co/IlyaGusev/mbart_ru_sum_gazeta) because it is trained for summarizing news in Russian and adapted to the domain of our data. Additionally, in the model author's article about the training dataset, you can see that the distribution of the number of tokens per sentence in the test set and the model's output is suitable for our task. [arxiv:2006.11063](https://arxiv.org/pdf/2006.11063.pdf)


## Classification 




<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ## Feature 3: Activity recognition on images 
  
If I haven't filled out the report, but attached photos, bot automatically analyzes the images and recognizes activities -->


<!-- 

### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url] -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->


<!-- ROADMAP -->
<!-- ## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



## Project structure

The project has the following structure:
- `news_monitoring/eda`: clustering scripts  
- `news_monitoring/models`: `.py` scripts with summarization and classification models
- `news_monitoring/preprocessing`: `.py` scripts with text preprocessing 
- `news_monitoring/preprocessing/news_monitoring.ipynb`: inference notebook 



<!-- ROADMAP -->
## Roadmap

- [x] Topic modeling 
- [x] News summarizing
- [x] News classificator

- [ ] News deduplication
- [ ] App for news scrapping





<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contacts

Telegram: [@my_name_is_nikita_hey](https://t.me/my_name_is_nikita_hey) <br>
Mail: tttonyalpha@gmail.com 



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/channel_screen.png
[lstm_predictions]: images/lstm_predictions.png
[lstm_recsys]: images/lstm_recsys.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

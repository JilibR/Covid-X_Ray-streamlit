import streamlit as st

def display_intro():

    st.markdown("""
    L’intelligence artificielle révolutionne aujourd’hui le domaine de la santé, notamment à travers l’automatisation et l’amélioration de l’interprétation des images médicales.
    **Ce projet s’inscrit à l’intersection de l’imagerie médicale et du Deep Learning**, avec pour objectif de concevoir un modèle capable d’analyser des radiographies pulmonaires, afin d’accélérer le diagnostic et de réduire la charge de travail des professionnels de santé.
    """)

    st.image("https://offices-appines-prod.s3.eu-west-1.amazonaws.com/60b8e6c746560a0152654dbe-64b501c5cce02a35918af8da-sd.png",
                 caption="Exemple de radiographie pulmonaire analysée par le modèle", use_container_width=True)

    st.markdown("""
    ---

    **Contexte et enjeux**
    En période de crise sanitaire, comme celle de la COVID-19, la pression sur les systèmes de santé est considérable.
    Les radiologues et médecins doivent traiter un volume élevé d’examens, souvent dans des délais très courts.
    L’automatisation de l’analyse des radiographies pulmonaires représente une opportunité majeure pour :
    - **Optimiser le temps médical** en réduisant le temps d’examen nécessaire,
    - **Améliorer la rapidité et la précision des diagnostics**, notamment dans les régions où l’accès aux soins est limité,
    - **Faciliter le tri des cas urgents** et réduire les erreurs liées à la surcharge de travail.
    """)
    try:
        image = Image.open("images/nom_de_ton_image.jpg")
        st.image(image, caption="Exemple de radiographie pulmonaire analysée par le modèle", use_column_width=True)
    except:
        # Option 2 : Image depuis une URL (exemple)
        st.image("https://marketing.webassets.siemens-healthineers.com/1800000005363926/a4e0eb057583/v/776ec3a71434/SIEMENS_HEALTHINEERS_Ysio_Max_Xray_Image_1800000005363926.jpg",
                 caption="Exemple de radiographie pulmonaire analysée par le modèle", use_container_width=True)

    st.markdown("""
    ---

    **Approche technique**
    Le projet repose sur une méthodologie rigoureuse de Deep Learning, combinant **classification** et **segmentation** :
    - **Classification** : identifier automatiquement la présence de pathologies (COVID-19, pneumonie, opacité pulmonaire, etc.) à partir d’une radiographie,
    - **Segmentation** : générer un masque binaire des poumons pour localiser précisément les zones affectées.

    Pour ce faire, nous utilisons un jeu de données varié et annoté, issu de la plateforme Kaggle (COVID-19 Radiography Database), comprenant plus de 21 000 images de radiographies pulmonaires, couvrant plusieurs pathologies.
    Ces données, bien que précieuses, soulèvent des questions éthiques et réglementaires, notamment en Europe avec le RGPD, qui encadre strictement l’utilisation des données de santé.

    ---

    **Impact et perspectives**
    Au-delà de l’aspect technique, ce projet vise à **démocratiser l’accès à un outil de diagnostic rapide et fiable**, particulièrement utile dans les zones rurales ou les pays en développement.
    L’intégration du modèle dans les systèmes hospitaliers, via une interface intuitive, permettrait aux praticiens de l’utiliser directement pour un diagnostic assisté.

    Enfin, ce travail s’inscrit dans une démarche de recherche appliquée, cherchant à concilier **innovation technologique, éthique et viabilité économique**, pour une médecine plus accessible et performante.
    """)
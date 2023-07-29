# Modelagem de processo de negócio
O fluxo de atividades do principal processo de negócio do nosso produto é a análise de dados. Quando o usuário acessa a página da funcionalidade o fluxo do processo de análise de dados inicia-se, sendo suas principais atividades e relações as seguintes:
1. Adicionar dataset
2. Dataset está com formato e tamanho válidos?
   2.1. Se sim, o processo continua.
   2.2 Se não, o usuário precisará adicionar outro dataset enquanto ele não estiver adequado.
3. O usuário selecionará o processamento dos dados e os algoritmos que deseja trabalhar.
4. Em seguida, ele analisa o dataset e o processo é concluído.
5. Caso a análise esteja concluída o processo é encerrado. E caso a análise não esteja satisfatória, o usuário poderá selecionar novos processamentos ou novas análises do dataset.
<img src="https://github.com/4Banks/documentation/blob/main/images/bpmn.jpeg" alt="BPMN" width="1200" height="400">

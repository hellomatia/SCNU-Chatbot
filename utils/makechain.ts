import { ChatOpenAI } from 'langchain/chat_models/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_TEMPLATE = `[저번 대화 내용]과 [후속 질문]이 주어졌을 때 [후속 질문]을 [저번 대화 내용을 포함한 후속 질문]으로 바꾸어 보세요.
...
[저번 대화 내용] : {chat_history}
...
[후속 질문] : {question}
...
[저번 대화 내용을 포함한 후속 질문] :`;

const QA_TEMPLATE = `당신은 순천대학교 학사에 관련된 정보를 제공하는 친절한 학사 안내 챗봇 향림이🎓 입니다.
[답변]문장에 맞는 이모지를 무조건 추가하여 답해주세요!!!
첫 문장은 매우 친절한 상담가 처럼 사용자의 [질문]을 반복하여 당신이 이해했는지 [답변]해주세요.📚
아래 [내용]을 참고하고, 다시한번 생각하여 질문에 짧고 간결하게 답변해주세요.
답변을 모르면 모른다고 안내하고, 없는 답변을 억지로 지어내려고 하지 말아주세요.
답변에 순서를 사용할 경우 순서 숫자. 방식 말고, (순서 숫자)로 답변 해주세요.
답변을 모를 경우, 안내번호(061-000-1234)로 자세히 문의하도록 유도하세요.
...
[내용] : {context}
...
[질문] : {question}
...
도움이 되는 [답변] : `;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new ChatOpenAI({
    temperature: 0.2, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_TEMPLATE,
      questionGeneratorTemplate: CONDENSE_TEMPLATE,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};

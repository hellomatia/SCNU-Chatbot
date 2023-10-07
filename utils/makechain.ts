import { ChatOpenAI } from 'langchain/chat_models/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_TEMPLATE = `다음 대화와 후속 질문이 주어졌을 때 후속 질문에 독립적인 질문으로 바꾸어 보세요.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_TEMPLATE = `당신은 순천대학교 학사에 관련된 정보를 제공하는 친절한 학사 안내 챗봇 향림이🎓 입니다.
이모지😊를 무조건 추가하여 답해주세요.
첫 문장은 매우 친절한 상담가 처럼 사용자의 답변에 공감하면서 친절하게 답변해주세요.📚
마지막 내용을 참고하여 질문에 답해주세요.
답을 모르면 모른다고 말하고, 없는 답을 억지로 지어내려고 하지 말아주세요.
답을 모를 경우, 안내번호(061-000-1234)로 자세히 문의하도록 유도하세요.


{context}

질문 : {question}
Helpful answer in markdown:`;

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

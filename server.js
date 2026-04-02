require('dotenv').config();
const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT_DIR = __dirname;
loadEnvFile();

const PORT = Number(process.env.PORT) || 3001;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const STATIC_ROUTES = {
  "/": "index.html",
  "/index.html": "index.html",
  "/chat": "chat.html",
  "/chat.html": "chat.html",
};

const CONTENT_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".ico": "image/x-icon",
};

const server = http.createServer(async (req, res) => {
  try {
    const requestUrl = new URL(req.url, `http://${req.headers.host || "localhost"}`);
    const { pathname } = requestUrl;

    if (req.method === "GET" && pathname === "/api/health") {
      return sendJson(res, 200, {
        status: "ok",
        apiConfigured: Boolean(process.env.OPENAI_API_KEY),
        model: OPENAI_MODEL,
      });
    }

    if (req.method === "POST" && pathname === "/api/ask") {
      const body = await readJsonBody(req);
      const question = typeof body.question === "string" ? body.question.trim() : "";
      const topic = normalizeTopic(body.topic);

      if (!question) {
        return sendJson(res, 400, {
          error: "Please enter a health-related question before submitting.",
        });
      }

      if (question.length < 5) {
        return sendJson(res, 400, {
          error: "Please enter a little more detail so the AI can respond helpfully.",
        });
      }

      if (question.length > 500) {
        return sendJson(res, 400, {
          error: "Please keep the question under 500 characters.",
        });
      }

      if (!looksHealthRelated(question)) {
        return sendJson(res, 200, {
          ...outOfScopeResponse(),
          topic,
          model: OPENAI_MODEL,
          inScope: false,
        });
      }

      const result = await callOpenAI(question, topic);
      return sendJson(res, 200, {
        ...result,
        topic,
        model: OPENAI_MODEL,
      });
    }

    if (req.method === "POST" && pathname === "/api/retina-scan") {
      // Allow larger payloads for retina image uploads (base64 JSON payloads can be several MB)
      const body = await readJsonBody(req, 15_000_000);
      const image = typeof body.image === "string" ? body.image.trim() : "";
      const question = typeof body.question === "string" ? body.question.trim() : "";

      if (!image) {
        return sendJson(res, 400, {
          error: "Please upload a retinal fundus image before submitting.",
        });
      }

      const retinaResult = await callRetinaModel(image);

      let educationalInfo = buildRetinaFallbackGuidance(retinaResult);

      if (question && looksHealthRelated(question) && process.env.OPENAI_API_KEY) {
        try {
          const prompt = `A retinal screening model returned:
        Prediction: ${retinaResult.prediction}
        Confidence: ${retinaResult.confidenceDisplay || retinaResult.confidence}
        Suggested guidance: ${retinaResult.suggestion}
        Screening summary: ${retinaResult.presentationInfo?.screeningSummary || "Not provided"}
        Prototype note: ${retinaResult.presentationInfo?.prototypeNote || "Not provided"}
        Situation summary: ${retinaResult.patientGuidance?.situation || "Not provided"}
        Why this may matter: ${retinaResult.patientGuidance?.whyItMatters || "Not provided"}
        Follow-up urgency: ${retinaResult.patientGuidance?.urgency || "routine"}
        What can be done: ${Array.isArray(retinaResult.patientGuidance?.whatCanBeDone) ? retinaResult.patientGuidance.whatCanBeDone.join("; ") : "Not provided"}
        Monitoring tips: ${Array.isArray(retinaResult.presentationInfo?.monitoringTips) ? retinaResult.presentationInfo.monitoringTips.join("; ") : "Not provided"}
        Questions to ask: ${Array.isArray(retinaResult.presentationInfo?.questionsToAsk) ? retinaResult.presentationInfo.questionsToAsk.join("; ") : "Not provided"}
        Warning signs: ${Array.isArray(retinaResult.patientGuidance?.warningSigns) ? retinaResult.patientGuidance.warningSigns.join("; ") : "Not provided"}

        User question: ${question}

        Give a short educational explanation of what this screening result may mean, what general next steps people often consider, and when to seek professional eye care. Do not diagnose.`;
          const aiGuidance = await callOpenAI(prompt, "eye-health");
          educationalInfo = {
            ...educationalInfo,
            ...aiGuidance,
            urgency: mergeUrgency(educationalInfo.urgency, aiGuidance.urgency),
          };
        } catch (_error) {}
      }

      return sendJson(res, 200, {
        mode: "retina-scan",
        retina: retinaResult,
        guidance: educationalInfo,
      });
    }

    if (req.method === "GET" && STATIC_ROUTES[pathname]) {
      return serveFile(res, path.join(ROOT_DIR, STATIC_ROUTES[pathname]));
    }

    if (req.method === "GET" && pathname === "/favicon.ico") {
      res.writeHead(204);
      res.end();
      return;
    }


    // Attempt to serve static assets directly from the project directory (css, js, images, etc.)
    if (req.method === "GET") {
      const assetRelative = pathname.replace(/^\/+/, "");
      const assetPath = path.join(ROOT_DIR, assetRelative);
      const resolvedAsset = path.resolve(assetPath);
      const resolvedRoot = path.resolve(ROOT_DIR);

      if (
        resolvedAsset.startsWith(resolvedRoot) &&
        fs.existsSync(resolvedAsset) &&
        fs.statSync(resolvedAsset).isFile()
      ) {
        return serveFile(res, resolvedAsset);
      }
    }

    sendJson(res, 404, { error: "Route not found." });
  } catch (error) {
    const statusCode = error.statusCode || 500;
    const message =
      statusCode >= 500
        ? error.publicMessage || "The server could not complete the request."
        : error.message;

    console.error("[server]", error);
    sendJson(res, statusCode, { error: message });
  }
});

server.listen(PORT, () => {
  console.log(`AIcura server running at http://localhost:${PORT}`);
});

function loadEnvFile() {
  const envPath = path.join(ROOT_DIR, ".env");

  if (!fs.existsSync(envPath)) {
    return;
  }

  const envContent = fs.readFileSync(envPath, "utf8");

  for (const rawLine of envContent.split(/\r?\n/)) {
    const line = rawLine.trim();

    if (!line || line.startsWith("#")) {
      continue;
    }

    const separatorIndex = line.indexOf("=");

    if (separatorIndex === -1) {
      continue;
    }

    const key = line.slice(0, separatorIndex).trim();
    const value = line.slice(separatorIndex + 1).trim().replace(/^"(.*)"$/, "$1");

    if (key && !(key in process.env)) {
      process.env[key] = value;
    }
  }
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(payload));
}

function serveFile(res, filePath) {
  fs.readFile(filePath, (error, fileBuffer) => {
    if (error) {
      sendJson(res, 404, { error: "File not found." });
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      "Content-Type": CONTENT_TYPES[ext] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(fileBuffer);
  });
}

function readJsonBody(req, maxChars = 1_000_000) {
  return new Promise((resolve, reject) => {
    let rawBody = "";
    let finished = false;

    function cleanup() {
      req.removeListener("data", onData);
      req.removeListener("end", onEnd);
      req.removeListener("error", onError);
      req.removeListener("close", onClose);
    }

    function onData(chunk) {
      if (finished) return;
      rawBody += chunk;

      if (rawBody.length > maxChars) {
        finished = true;
        cleanup();
        const error = new Error("Request body is too large.");
        error.statusCode = 413;
        reject(error);
        return;
      }
    }

    function onEnd() {
      if (finished) return;
      finished = true;
      cleanup();

      if (!rawBody) {
        resolve({});
        return;
      }

      try {
        resolve(JSON.parse(rawBody));
      } catch (_error) {
        const error = new Error("Invalid JSON body.");
        error.statusCode = 400;
        reject(error);
      }
    }

    function onError(err) {
      if (finished) return;
      finished = true;
      cleanup();
      reject(err);
    }

    function onClose() {
      if (finished) return;
      finished = true;
      cleanup();
      const error = new Error("Connection closed before body was fully received.");
      error.statusCode = 499;
      reject(error);
    }

    req.on("data", onData);
    req.on("end", onEnd);
    req.on("error", onError);
    req.on("close", onClose);
  });
}

function normalizeTopic(topic) {
  return topic === "eye-health" ? "eye-health" : "general-health";
}

async function callOpenAI(question, topic) {
  if (!process.env.OPENAI_API_KEY) {
    const error = new Error("OPENAI_API_KEY is missing. Add it to your .env file before calling the AI API.");
    error.statusCode = 500;
    error.publicMessage = error.message;
    throw error;
  }

  const systemPrompt = buildSystemPrompt(topic);
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      temperature: 0.3,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: question },
      ],
    }),
  });

  const responseBody = await response.json();

  if (!response.ok) {
    const error = new Error(responseBody.error?.message || "The AI API request failed.");
    error.statusCode = response.status === 429 ? 503 : 502;
    error.publicMessage = "The AI service is unavailable right now. Please try again.";
    throw error;
  }

  const messageContent = responseBody.choices?.[0]?.message?.content;

  if (!messageContent) {
    const error = new Error("The AI API returned an empty response.");
    error.statusCode = 502;
    error.publicMessage = "The AI service returned an empty response. Please try again.";
    throw error;
  }

  let parsedResult;

  try {
    parsedResult = JSON.parse(stripCodeFence(messageContent));
  } catch (_error) {
    if (!looksHealthRelated(question)) {
      return {
        ...outOfScopeResponse(),
        inScope: false,
      };
    }

    parsedResult = {
      answer: messageContent.trim(),
      explanation: fallbackExplanation(topic),
      disclaimer: fallbackDisclaimer(),
      urgency: "routine",
      in_scope: true,
    };
  }

  const inScope = normalizeInScope(parsedResult.in_scope);

  if (!inScope) {
    return {
      ...outOfScopeResponse(),
      inScope: false,
    };
  }

  return {
    answer: cleanText(parsedResult.answer) || "The AI service did not return an answer.",
    explanation: cleanText(parsedResult.explanation) || fallbackExplanation(topic),
    disclaimer: cleanText(parsedResult.disclaimer) || fallbackDisclaimer(),
    urgency: normalizeUrgency(parsedResult.urgency),
    inScope: true,
  };
}

function buildSystemPrompt(topic) {
  const scope =
    topic === "eye-health"
      ? "Prioritize eye and vision health context when relevant."
      : "Provide broad educational health information in plain language.";

  return [
    "You are AIcura, an educational healthcare information assistant.",
    scope,
    "You only answer health-related questions.",
    "Allowed topics include symptoms, wellness, preventive care, medications, general medical information, eye health, retinal scans, heart health, diabetes, and medical safety guidance.",
    "If a request is not about health, medical care, wellness, retinal scans, eye health, heart health, diabetes, or a closely related health topic, do not answer it.",
    'For out-of-scope requests, return JSON with "in_scope": false and tell the user that AIcura only answers health-related questions.',
    "Do not diagnose, prescribe treatment, confirm a condition, or present yourself as a medical professional.",
    "Never state or imply that the user definitely has a condition.",
    'Use careful phrasing such as "possible explanations include", "this can sometimes be associated with", or "people may experience this when".',
    "If symptoms suggest an emergency, give general safety guidance to seek urgent medical attention immediately.",
    "Keep the answer concise, practical, and understandable for non-experts.",
    'Return only valid JSON with the keys "in_scope", "answer", "explanation", "disclaimer", and "urgency".',
    'Use one of these urgency values exactly: "routine", "soon", "urgent". The urgency field is general care guidance only, not a diagnosis.',
    'The "answer" should be 2-4 sentences.',
    'The "explanation" should be 1-2 sentences about what factors matter.',
    'The "disclaimer" must clearly say the response is for educational purposes only, is not medical advice, and is not a diagnosis.',
  ].join(" ");
}

function stripCodeFence(text) {
  return text.replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/\s*```$/, "").trim();
}

function cleanText(value) {
  return typeof value === "string" ? value.replace(/\s+/g, " ").trim() : "";
}

function normalizeUrgency(value) {
  return ["routine", "soon", "urgent"].includes(value) ? value : "routine";
}

function normalizeInScope(value) {
  if (typeof value === "string") {
    return value.toLowerCase() !== "false";
  }

  return value !== false;
}

function looksHealthRelated(question) {
  const normalizedQuestion = question.toLowerCase();
  const healthKeywords = [
    "health",
    "medical",
    "medicine",
    "doctor",
    "clinic",
    "hospital",
    "symptom",
    "symptoms",
    "pain",
    "fever",
    "cough",
    "cold",
    "flu",
    "headache",
    "migraine",
    "dizzy",
    "dizziness",
    "nausea",
    "vomit",
    "diarrhea",
    "constipation",
    "rash",
    "swelling",
    "infection",
    "injury",
    "wound",
    "fatigue",
    "tired",
    "sleep",
    "stress",
    "anxiety",
    "depression",
    "mental health",
    "therapy",
    "medication",
    "medicine",
    "pharmacy",
    "dose",
    "supplement",
    "vitamin",
    "diet",
    "nutrition",
    "exercise",
    "hydration",
    "allergy",
    "asthma",
    "breathing",
    "shortness of breath",
    "blood pressure",
    "cholesterol",
    "heart",
    "cardiac",
    "diabetes",
    "glucose",
    "insulin",
    "retina",
    "retinal",
    "eye",
    "vision",
    "blurry",
    "blurred",
    "dry eyes",
    "red eye",
    "fundus",
    "screening",
    "treatment",
    "wellness",
    "throat",
    "chest",
    "stomach",
    "abdomen",
    "skin",
    "back pain",
    "joint",
    "pregnancy",
  ];

  return healthKeywords.some((keyword) => normalizedQuestion.includes(keyword));
}

function outOfScopeResponse() {
  return {
    answer: "AIcura only answers health-related questions. Please ask about symptoms, wellness, eye or retinal health, heart health, diabetes, or other medical topics.",
    explanation: "Your message did not appear to be about a health-related topic, so no health guidance was generated.",
    disclaimer: fallbackDisclaimer(),
    urgency: "routine",
  };
}

function fallbackExplanation(topic) {
  if (topic === "eye-health") {
    return "This summary highlights common eye-health considerations related to your question and should be read as general educational information.";
  }

  return "This summary focuses on common health information related to your question and should be read as general educational guidance.";
}

function fallbackDisclaimer() {
  return "For educational purposes only. This AI-generated response is not medical advice, is not a diagnosis, and does not replace a licensed healthcare professional.";
}

function buildRetinaFallbackGuidance(retinaResult) {
  const patientGuidance = retinaResult?.patientGuidance || {};
  const presentationInfo = retinaResult?.presentationInfo || {};
  const actionText = Array.isArray(patientGuidance.whatCanBeDone)
    ? patientGuidance.whatCanBeDone.map(cleanText).filter(Boolean).join(" ")
    : "";
  const monitoringText = Array.isArray(presentationInfo.monitoringTips)
    ? presentationInfo.monitoringTips.map(cleanText).filter(Boolean).join(" ")
    : "";

  const explanation = [
    cleanText(presentationInfo.educationalContext),
    cleanText(patientGuidance.whyItMatters),
    cleanText(patientGuidance.followUp),
    actionText,
    monitoringText,
  ]
    .filter(Boolean)
    .join(" ");

  return {
    answer:
      cleanText(presentationInfo.screeningSummary) ||
      cleanText(patientGuidance.situation) ||
      "This result is based on the uploaded retinal image.",
    explanation:
      explanation ||
      "Retinal screening tools can sometimes highlight patterns that may need follow-up with an eye specialist.",
    disclaimer: cleanText(retinaResult?.disclaimer) || fallbackDisclaimer(),
    urgency: normalizeUrgency(patientGuidance.urgency),
  };
}

function mergeUrgency(...values) {
  const urgencyRank = {
    routine: 0,
    soon: 1,
    urgent: 2,
  };

  return values.reduce((highest, current) => {
    const normalizedCurrent = normalizeUrgency(current);
    return urgencyRank[normalizedCurrent] > urgencyRank[highest] ? normalizedCurrent : highest;
  }, "routine");
}

async function callRetinaModel(imageBase64) {
  let response;

  try {
    response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: imageBase64,
      }),
    });
  } catch (_error) {
    const error = new Error("The retina analysis service is unavailable right now. Please try again.");
    error.statusCode = 502;
    error.publicMessage = error.message;
    throw error;
  }

  let responseBody = {};

  try {
    responseBody = await response.json();
  } catch (_error) {
    responseBody = {};
  }

  if (!response.ok) {
    const error = new Error(responseBody.error || "Retina model request failed.");

    if (response.status >= 500) {
      error.statusCode = 502;
      error.publicMessage = "The retina analysis service is unavailable right now. Please try again.";
    } else {
      error.statusCode = response.status;
    }

    throw error;
  }

  return responseBody;
}

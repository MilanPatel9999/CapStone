const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT_DIR = __dirname;
loadEnvFile();

const PORT = Number(process.env.PORT) || 3000;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const RETFOUND_SERVICE_URL = process.env.RETFOUND_SERVICE_URL || "http://127.0.0.1:8001";
const RETFOUND_REQUEST_TIMEOUT_MS = Number(process.env.RETFOUND_REQUEST_TIMEOUT_MS) || 90_000;
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
      const retinalService = await getRetinalServiceHealth();
      return sendJson(res, 200, {
        status: "ok",
        apiConfigured: Boolean(process.env.OPENAI_API_KEY),
        model: OPENAI_MODEL,
        retinalService,
      });
    }

    if (req.method === "POST" && pathname === "/api/ask") {
      const body = await readJsonBody(req);
      const question = typeof body.question === "string" ? body.question.trim() : "";
      const topic = normalizeTopic(body.topic);
      const retinalImage = normalizeRetinalImage(body.retinalImage);
      const hasRetinalWorkflowInput = topic === "eye-health" && Boolean(retinalImage);

      if (!question && !hasRetinalWorkflowInput) {
        return sendJson(res, 400, {
          error: "Please enter a health-related question or upload a retinal scan image before submitting.",
        });
      }

      if (question && question.length < 5 && !hasRetinalWorkflowInput) {
        return sendJson(res, 400, {
          error: "Please enter a little more detail so the AI can respond helpfully.",
        });
      }

      if (question.length > 500) {
        return sendJson(res, 400, {
          error: "Please keep the question under 500 characters.",
        });
      }

      const effectiveQuestion = buildEffectiveQuestion(question, topic, retinalImage);

      if (!hasRetinalWorkflowInput && !looksHealthRelated(effectiveQuestion)) {
        return sendJson(res, 200, {
          ...outOfScopeResponse(),
          topic,
          model: OPENAI_MODEL,
          inScope: false,
        });
      }

      const retinalAnalysis =
        hasRetinalWorkflowInput
          ? await callRetinalService(effectiveQuestion, topic, retinalImage)
          : null;
      if (retinalAnalysis && retinalAnalysis.isRetinalImage === false) {
        return sendJson(res, 200, {
          ...buildInvalidRetinalImageResponse(),
          topic,
          model: OPENAI_MODEL,
          inScope: true,
          retinalAnalysis,
        });
      }

      const aiQuestion = buildQuestionWithRetinalContext(effectiveQuestion, topic, retinalAnalysis);
      const result = await callOpenAI(aiQuestion, topic);

      return sendJson(res, 200, {
        ...result,
        topic,
        model: OPENAI_MODEL,
        retinalAnalysis,
      });
    }

    if (req.method === "GET" && STATIC_ROUTES[pathname]) {
      return serveFile(res, path.join(ROOT_DIR, STATIC_ROUTES[pathname]));
    }

    if (req.method === "GET") {
      const staticFilePath = resolveStaticFilePath(pathname);

      if (staticFilePath) {
        return serveFile(res, staticFilePath);
      }
    }

    if (req.method === "GET" && pathname === "/favicon.ico") {
      res.writeHead(204);
      res.end();
      return;
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

function resolveStaticFilePath(requestPath) {
  const decodedPath = decodeURIComponent(requestPath || "/");
  const relativePath = decodedPath.replace(/^\/+/, "");

  if (!relativePath) {
    return null;
  }

  const resolvedPath = path.resolve(ROOT_DIR, relativePath);

  if (!resolvedPath.startsWith(ROOT_DIR)) {
    return null;
  }

  if (!fs.existsSync(resolvedPath)) {
    return null;
  }

  const fileStats = fs.statSync(resolvedPath);
  return fileStats.isFile() ? resolvedPath : null;
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let rawBody = "";

    req.on("data", (chunk) => {
      rawBody += chunk;

      if (rawBody.length > 10_000_000) {
        const error = new Error("Request body is too large.");
        error.statusCode = 413;
        reject(error);
        req.destroy();
      }
    });

    req.on("end", () => {
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
    });

    req.on("error", (error) => {
      reject(error);
    });
  });
}

function normalizeTopic(topic) {
  return topic === "eye-health" ? "eye-health" : "general-health";
}

function buildEffectiveQuestion(question, topic, retinalImage) {
  const trimmedQuestion = typeof question === "string" ? question.trim() : "";

  if (topic !== "eye-health" || !retinalImage) {
    return trimmedQuestion;
  }

  const retinalContext =
    'The user uploaded an image in retinal scan mode. Respond only with educational, non-diagnostic information about retinal scans, eye health, heart-health related retinal patterns, diabetes-related retinal patterns, or safe next steps. If the typed question is vague, interpret it as a request for educational retinal-image guidance.';

  if (!trimmedQuestion) {
    return `${retinalContext} Explain what a retinal-analysis workflow may look for in an uploaded scan and remind the user that image-based output is educational only.`;
  }

  return `${trimmedQuestion}\n\n${retinalContext}`;
}

function normalizeRetinalImage(retinalImage) {
  if (!retinalImage || typeof retinalImage !== "object") {
    return null;
  }

  const name = typeof retinalImage.name === "string" ? retinalImage.name.trim() : "";
  const mimeType = typeof retinalImage.mimeType === "string" ? retinalImage.mimeType.trim() : "";
  const dataUrl = typeof retinalImage.dataUrl === "string" ? retinalImage.dataUrl.trim() : "";

  if (!dataUrl) {
    return null;
  }

  if (!/^data:image\/(png|jpeg|jpg|webp);base64,/i.test(dataUrl)) {
    return null;
  }

  if (dataUrl.length > 8_000_000) {
    const error = new Error("Please upload a retinal image under roughly 6 MB.");
    error.statusCode = 413;
    throw error;
  }

  return {
    name: name || "uploaded-retinal-image",
    mimeType: mimeType || "image/jpeg",
    dataUrl,
  };
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
    "If the user message includes retinal-model context, treat it as an educational screening signal only and never as a confirmed finding.",
    "If a retinal model output label is provided, do not say the person has that disease or stage. Instead say the model flagged a pattern or produced a screening label that would still need clinician review.",
    "Never restate diabetic retinopathy, heart risk, or any retinal-model output as a diagnosis.",
    "If symptoms suggest an emergency, give general safety guidance to seek urgent medical attention immediately.",
    "Keep the answer concise, practical, and understandable for non-experts.",
    'Return only valid JSON with the keys "in_scope", "answer", "explanation", "disclaimer", and "urgency".',
    'Use one of these urgency values exactly: "routine", "soon", "urgent". The urgency field is general care guidance only, not a diagnosis.',
    'The "answer" should be 2-4 sentences.',
    'The "explanation" should be 1-2 sentences about what factors matter.',
    'The "disclaimer" must clearly say the response is for educational purposes only, is not medical advice, and is not a diagnosis.',
  ].join(" ");
}

function buildQuestionWithRetinalContext(question, topic, retinalAnalysis) {
  if (topic !== "eye-health" || !retinalAnalysis || retinalAnalysis.isRetinalImage === false) {
    return question;
  }

  const contextParts = [
    "Retinal analysis context:",
    retinalAnalysis.summary || "",
  ];

  if (retinalAnalysis.validation) {
    contextParts.push(
      `Validation accepted: ${retinalAnalysis.validation.accepted ? "yes" : "no"}.`,
      `Retinal confidence: ${retinalAnalysis.validation.retinalScore ?? "unavailable"}.`,
    );
  }

  if (retinalAnalysis.diabetesAnalysis?.topPrediction) {
    const diabetesSignalGuidance = describeScreeningSignalGuidance(
      "diabetes",
      retinalAnalysis.diabetesAnalysis.topPrediction,
      retinalAnalysis.diabetesAnalysis.confidenceBand,
    );
    contextParts.push(...diabetesSignalGuidance);
  }

  if (retinalAnalysis.heartAnalysis?.topPrediction) {
    const heartSignalGuidance = describeScreeningSignalGuidance(
      "heart",
      retinalAnalysis.heartAnalysis.topPrediction,
      retinalAnalysis.heartAnalysis.confidenceBand,
    );
    contextParts.push(...heartSignalGuidance);
  } else if (retinalAnalysis.heartAnalysis?.status === "pending_checkpoint") {
    contextParts.push(
      "Heart-specific RETFound checkpoint is not configured yet.",
      "Do not describe any heart-risk result for this upload because no heart model output exists.",
    );
  }

  return `${question}\n\n${contextParts.join(" ")}`;
}

function describeScreeningSignalGuidance(domain, prediction, confidenceBand) {
  if (!prediction) {
    return [];
  }

  const label = prediction.label;
  const confidence = prediction.confidence;

  if (confidenceBand === "low") {
    return [
      `${capitalize(domain)} screening output was low confidence: top label ${label} at ${confidence}.`,
      `Do not restate "${label}" as a likely ${domain} stage or finding in the user-facing answer.`,
      "Say only that the screening branch produced an uncertain result that may warrant professional review.",
    ];
  }

  if (confidenceBand === "tentative") {
    return [
      `${capitalize(domain)} screening output was tentative: top label ${label} at ${confidence}.`,
      `Do not say the user has a confirmed ${domain} condition or stage.`,
      "Present it only as a cautious screening signal that still needs clinician review.",
    ];
  }

  return [
    `${capitalize(domain)} screening output label (not a diagnosis): ${label} at ${confidence}.`,
    `Do not say the user has a confirmed ${domain} condition or stage.`,
    "Present it only as a screening signal that may warrant professional review.",
  ];
}

function capitalize(value) {
  if (!value) {
    return "";
  }

  return value.charAt(0).toUpperCase() + value.slice(1);
}

async function getRetinalServiceHealth() {
  try {
    const response = await fetch(`${RETFOUND_SERVICE_URL}/health`, {
      signal: AbortSignal.timeout(1200),
    });

    if (!response.ok) {
      throw new Error(`Retinal service health check failed with status ${response.status}.`);
    }

    const data = await response.json();
    return {
      configured: Boolean(RETFOUND_SERVICE_URL),
      reachable: true,
      requestedMode: data.requestedMode || "demo",
      activeMode: data.activeMode || "demo",
      modelLoaded: Boolean(data.modelLoaded),
      modelName: data.modelName || "",
      validatorEnabled: Boolean(data.validatorEnabled),
      diabetesEnabled: Boolean(data.diabetesEnabled),
      diabetesModelLoaded: Boolean(data.diabetesModelLoaded),
    };
  } catch (_error) {
    return {
      configured: Boolean(RETFOUND_SERVICE_URL),
      reachable: false,
      requestedMode: "unknown",
      activeMode: "unavailable",
      modelLoaded: false,
      modelName: "",
      validatorEnabled: false,
      diabetesEnabled: false,
      diabetesModelLoaded: false,
    };
  }
}

async function callRetinalService(question, topic, retinalImage) {
  try {
    const response = await fetch(`${RETFOUND_SERVICE_URL}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(RETFOUND_REQUEST_TIMEOUT_MS),
      body: JSON.stringify({
        question,
        topic,
        image_name: retinalImage.name,
        image_data_url: retinalImage.dataUrl,
      }),
    });

    const rawText = await response.text();
    let data = null;

    if (rawText) {
      try {
        data = JSON.parse(rawText);
      } catch (_error) {
        throw new Error(rawText || "Retinal image analysis failed.");
      }
    }

    if (!response.ok) {
      throw new Error(data?.detail || data?.error || "Retinal image analysis failed.");
    }

    return data;
  } catch (error) {
    return buildRetinalServiceFallback(error);
  }
}

function buildRetinalServiceFallback(error) {
  return {
    analyzerName: "RETFound service",
    requestedMode: "unknown",
    activeMode: "unavailable",
    modelLoaded: false,
    modelName: "",
    summary:
      "A retinal image was uploaded, but the retinal-analysis service could not be reached. The question-and-answer flow still completed without image inference.",
    imageProperties: null,
    qualityNotes: [],
    topPrediction: null,
    predictions: [],
    nextStep: "Start the Python retinal-analysis service and configure the model checkpoint before expecting image-based output.",
    disclaimer:
      "No retinal-model inference was produced for this request. Any answer shown here remains educational only and is not medical advice or a diagnosis.",
    serviceError: cleanText(error?.message || ""),
  };
}

function buildInvalidRetinalImageResponse() {
  return {
    answer:
      "This upload does not appear to be a retinal fundus image. Please upload a clear retinal scan or fundus photograph to use the retinal-analysis workflow.",
    explanation:
      "The retinal pipeline first validates whether the uploaded file looks like a real retinal image before running any retinal or diabetes analysis.",
    disclaimer:
      "For educational purposes only. This upload was rejected by the retinal-image validator, so no retinal disease or heart-risk analysis was performed.",
    urgency: "routine",
  };
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

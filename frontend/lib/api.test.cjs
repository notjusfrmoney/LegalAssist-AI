const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const typescript = require("typescript");

const source = readFileSync(path.join(__dirname, "api.ts"), "utf8");
const compiled = typescript.transpileModule(source, {
  compilerOptions: {
    module: typescript.ModuleKind.CommonJS,
    target: typescript.ScriptTarget.ES2022,
  },
}).outputText;

function loadApi(fetchImplementation) {
  const exports = {};
  new Function("exports", "require", "process", "fetch", compiled)(
    exports,
    require,
    process,
    fetchImplementation,
  );
  return exports;
}

test("structured FastAPI validation errors are readable", async (t) => {
  t.mock.method(console, "error", () => {});
  const api = loadApi(async () => ({
    ok: false,
    json: async () => ({
      detail: [
        {
          loc: ["body", "messages", 2, "role"],
          msg: "Input should be 'user' or 'assistant'",
        },
      ],
    }),
  }));

  await assert.rejects(
    api.sendChatMessage([{ role: "user", content: "My salary is pending" }]),
    (error) => {
      assert.equal(
        error.message,
        "body.messages.2.role: Input should be 'user' or 'assistant'",
      );
      assert.doesNotMatch(error.message, /\[object Object\]/);
      return true;
    },
  );
});

test("object-shaped FastAPI errors do not become object strings", async (t) => {
  t.mock.method(console, "error", () => {});
  const api = loadApi(async () => ({
    ok: false,
    json: async () => ({ detail: { message: "The assistant is temporarily unavailable." } }),
  }));

  await assert.rejects(
    api.sendChatMessage([{ role: "user", content: "My salary is pending" }]),
    (error) => {
      assert.equal(error.message, "The assistant is temporarily unavailable.");
      assert.doesNotMatch(error.message, /\[object Object\]/);
      return true;
    },
  );
});

test("string FastAPI errors are preserved", async (t) => {
  t.mock.method(console, "error", () => {});
  const api = loadApi(async () => ({
    ok: false,
    json: async () => ({ detail: "The answer service timed out. Please try again." }),
  }));

  await assert.rejects(
    api.sendChatMessage([{ role: "user", content: "My salary is pending" }]),
    (error) => {
      assert.equal(error.message, "The answer service timed out. Please try again.");
      return true;
    },
  );
});

test("network failures return a safe message", async (t) => {
  t.mock.method(console, "error", () => {});
  const api = loadApi(async () => {
    throw { cause: "connection failed" };
  });

  await assert.rejects(
    api.sendChatMessage([{ role: "user", content: "My salary is pending" }]),
    (error) => {
      assert.match(error.message, /Unable to connect/);
      assert.doesNotMatch(error.message, /\[object Object\]/);
      return true;
    },
  );
});

test("the displayed follow-up question is sent in assistant history", async () => {
  let requestBody;
  const api = loadApi(async (_url, options) => {
    requestBody = JSON.parse(options.body);
    return {
      ok: true,
      json: async () => ({ answer: "A grounded response.", citations: [] }),
    };
  });

  await api.sendChatMessage([
    { role: "user", content: "My salary is pending" },
    {
      role: "assistant",
      content: "I need one detail.",
      followUpQuestion: "Which month's salary is unpaid?",
    },
    { role: "user", content: "October's salary" },
  ]);

  assert.deepEqual(requestBody.messages, [
    { role: "user", content: "My salary is pending" },
    {
      role: "assistant",
      content: "I need one detail.\n\nFollow-up question: Which month's salary is unpaid?",
    },
    { role: "user", content: "October's salary" },
  ]);
});

test("unexpected errors never produce [object Object]", () => {
  const api = loadApi(async () => {});

  assert.doesNotMatch(api.getChatErrorMessage({ detail: {} }), /\[object Object\]/);
  assert.match(api.getChatErrorMessage({ detail: {} }), /could not respond/i);
});

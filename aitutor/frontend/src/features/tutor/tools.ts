import { FunctionDeclaration, Type } from "@google/genai";

export const PYTHON_SANDBOX_TOOL: FunctionDeclaration = {
    name: "run_python_code",
    description: "Executes Python code in a secure sandbox and returns the output (stdout/stderr). Use this for simulations, calculations, or data analysis.",
    parameters: {
        type: Type.OBJECT,
        properties: {
            code: {
                type: Type.STRING,
                description: "The Python code to execute.",
            },
        },
        required: ["code"],
    },
};

export const ALTAIR_TOOL: FunctionDeclaration = {
    name: "render_altair",
    description: "Displays an altair graph in json format.",
    parameters: {
        type: Type.OBJECT,
        properties: {
            json_graph: {
                type: Type.STRING,
                description:
                    "JSON STRING representation of the graph to render. Must be a string, not a json object",
            },
        },
        required: ["json_graph"],
    },
};

export const SOCRATIC_DEBUGGER_TOOL: FunctionDeclaration = {
    name: "draw_overlay",
    description: "Draws visual highlights or annotations on the screen/scratchpad. Use this to point out specific errors or guide the student's attention.",
    parameters: {
        type: Type.OBJECT,
        properties: {
            annotations: {
                type: Type.ARRAY,
                description: "List of annotations to draw.",
                items: {
                    type: Type.OBJECT,
                    properties: {
                        type: {
                            type: Type.STRING,
                            description: "Type of annotation: 'circle', 'arrow', 'text', 'box'",
                            enum: ["circle", "arrow", "text", "box"],
                        },
                        x: { type: Type.NUMBER, description: "X coordinate (0-100)" },
                        y: { type: Type.NUMBER, description: "Y coordinate (0-100)" },
                        width: { type: Type.NUMBER, description: "Width (0-100)" },
                        height: { type: Type.NUMBER, description: "Height (0-100)" },
                        label: { type: Type.STRING, description: "Optional text label" },
                        color: { type: Type.STRING, description: "Color (hex or name)" },
                    },
                    required: ["type", "x", "y"],
                },
            },
        },
        required: ["annotations"],
    },
};

export const KNOWLEDGE_RETRIEVAL_TOOL: FunctionDeclaration = {
    name: "search_knowledge_base",
    description: "Searches the educational knowledge base (Khan Academy, Wikipedia, etc.) for factual information.",
    parameters: {
        type: Type.OBJECT,
        properties: {
            query: {
                type: Type.STRING,
                description: "The search query.",
            },
        },
        required: ["query"],
    },
};

export const CONTEXTUAL_TELEPORTATION_TOOL: FunctionDeclaration = {
    name: "teleport_to_world",
    description: "Generates and teleports the student to a new 3D world related to the current topic.",
    parameters: {
        type: Type.OBJECT,
        properties: {
            prompt: {
                type: Type.STRING,
                description: "Description of the world to generate (e.g., 'Ancient Rome bathhouse').",
            },
        },
        required: ["prompt"],
    },
};

export const META_COGNITIVE_TOOL: FunctionDeclaration = {
    name: "log_thinking",
    description: "Logs the tutor's internal monologue about the student's progress and the chosen teaching strategy. This is visible to the teacher/observer.",
    parameters: {
        type: Type.OBJECT,
        properties: {
            thought: {
                type: Type.STRING,
                description: "The tutor's internal thought.",
            },
        },
        required: ["thought"],
    },
};

export const LESSON_PLANNER_TOOL: FunctionDeclaration = {
    name: "update_lesson_plan",
    description: "Updates the current lesson plan and marks milestones as completed.",
    parameters: {
        type: Type.OBJECT,
        properties: {
            milestones: {
                type: Type.ARRAY,
                description: "List of milestones with their status.",
                items: {
                    type: Type.OBJECT,
                    properties: {
                        label: { type: Type.STRING, description: "Milestone label" },
                        completed: { type: Type.BOOLEAN, description: "Whether it is completed" },
                        current: { type: Type.BOOLEAN, description: "Whether it is the current focus" },
                    },
                    required: ["label", "completed"],
                },
            },
        },
        required: ["milestones"],
    },
};

export const TUTOR_TOOLS = [
    PYTHON_SANDBOX_TOOL,
    ALTAIR_TOOL,
    SOCRATIC_DEBUGGER_TOOL,
    KNOWLEDGE_RETRIEVAL_TOOL,
    CONTEXTUAL_TELEPORTATION_TOOL,
    META_COGNITIVE_TOOL,
    LESSON_PLANNER_TOOL,
];

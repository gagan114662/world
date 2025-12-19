/**
 * Analysis Service - Interface for complex AI analysis features.
 */

import { apiUtils } from '../lib/api-utils';

const TA_SERVICE_URL = import.meta.env.VITE_TA_SERVICE_URL || 'http://localhost:8002';

export interface QuizQuestion {
    question: string;
    options: string[];
    correct_answer_index: number;
    explanation: string;
}

export interface AnalysisResult {
    knowledge_gaps: string[];
    quiz: QuizQuestion[];
}

export class AnalysisService {
    /**
     * Upload an audio explanation for analysis.
     * @param audioBlob The audio recording as a Blob.
     */
    static async uploadExplanation(audioBlob: Blob): Promise<AnalysisResult> {
        const formData = new FormData();
        formData.append('file', audioBlob, 'explanation.mp3');

        const response = await apiUtils.authenticatedFetch(`${TA_SERVICE_URL}/analysis/explain-to-learn`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Failed to analyze explanation' }));
            throw new Error(error.detail || 'Failed to analyze explanation');
        }

        return response.json();
    }
}

export default AnalysisService;

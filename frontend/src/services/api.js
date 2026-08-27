import axios from 'axios'

// In dev, Vite proxies /api -> the FastAPI backend (see vite.config.js).
// In production, set VITE_API_BASE_URL to the deployed backend URL.
const baseURL = import.meta.env.VITE_API_BASE_URL || '/api'

export const apiClient = axios.create({
  baseURL,
  timeout: 30_000,
})

/**
 * Normalizes backend errors into a consistent shape, whether they come from
 * our structured ErrorResponse envelope or a network-level failure.
 */
export function toApiError(error) {
  const backendError = error?.response?.data?.error
  if (backendError) {
    return {
      code: backendError.code || 'unknown_error',
      message: backendError.message || 'Something went wrong.',
      context: backendError.context ?? null,
      status: error.response.status,
    }
  }
  if (error?.request) {
    return {
      code: 'network_error',
      message: 'Could not reach the RepoLens backend. Is it running?',
      context: null,
      status: null,
    }
  }
  return {
    code: 'client_error',
    message: error?.message || 'An unexpected error occurred.',
    context: null,
    status: null,
  }
}

/** GET /api/health */
export async function getHealth() {
  const { data } = await apiClient.get('/health')
  return data
}

// Indexing (POST /api/index) and chat (POST /api/chat) helpers are added
// in later phases once those endpoints exist on the backend.

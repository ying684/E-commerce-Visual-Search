<template>
  <div class="min-h-screen bg-gray-50 text-gray-800 font-sans">
    <header class="bg-white shadow-sm py-6">
      <div class="max-w-7xl mx-auto px-4 flex items-center justify-center">
        <h1 class="text-3xl font-bold text-blue-600 tracking-tight">🛍️ E-commerce Visual Search</h1>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 py-10">
      <div class="max-w-2xl mx-auto bg-white rounded-xl shadow-md p-8 text-center border-2 border-dashed border-gray-300 hover:border-blue-500 transition-colors">
        <input 
          type="file" 
          ref="fileInput" 
          @change="onFileSelected" 
          accept="image/jpeg, image/png, image/webp"
          class="hidden"
        />
        
        <div v-if="!previewUrl" @click="$refs.fileInput.click()" class="cursor-pointer py-10">
          <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 48 48">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M24 14v20m-10-10h20" />
          </svg>
          <p class="mt-4 text-lg text-gray-600 font-medium">Nhấn vào đây để tải ảnh lên</p>
          <p class="text-sm text-gray-400 mt-1">Hỗ trợ JPG, PNG, WEBP</p>
        </div>

        <div v-else class="flex flex-col items-center">
          <img :src="previewUrl" class="h-64 object-contain rounded-lg shadow-sm mb-6 border border-gray-100" />
          <div class="flex gap-4">
            <button @click="$refs.fileInput.click()" class="px-5 py-2.5 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 font-medium transition">
              Đổi ảnh khác
            </button>
            <button @click="searchImage" :disabled="isLoading" class="px-6 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium flex items-center gap-2 disabled:opacity-70 transition shadow-sm hover:shadow">
              <span v-if="isLoading" class="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></span>
              {{ isLoading ? 'Đang quét AI Core...' : 'Tìm sản phẩm tương tự' }}
            </button>
          </div>
        </div>
      </div>

      <div v-if="errorMessage" class="max-w-2xl mx-auto mt-6 p-4 bg-red-100 border border-red-200 text-red-700 rounded-lg text-center font-medium">
        {{ errorMessage }}
      </div>

      <div v-if="results.length > 0" class="mt-16 animate-fade-in">
        <h2 class="text-2xl font-bold mb-6 border-b pb-3 text-gray-800">🎯 Top {{ results.length }} Sản phẩm tương đồng</h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6">
          
          <div v-for="item in results" :key="item.posting_id" class="bg-white rounded-xl shadow-sm hover:shadow-xl transition-all overflow-hidden border border-gray-100 group">
            <div class="h-56 w-full bg-white flex items-center justify-center p-2 border-b border-gray-50 overflow-hidden">
               <img :src="item.image_url" class="max-h-full object-contain group-hover:scale-105 transition-transform duration-300" />
            </div>
            
            <div class="p-4 bg-gray-50">
              <p class="text-xs text-gray-500 mb-2 truncate" :title="item.posting_id">Mã: {{ item.posting_id }}</p>
              <div class="flex items-center justify-between mt-2">
                <span class="text-sm font-bold text-blue-700 bg-blue-100 px-2.5 py-1 rounded-md">Top {{ item.rank }}</span>
                <span class="text-xs font-medium text-gray-500 bg-white px-2 py-1 border border-gray-200 rounded">Sai số: {{ item.distance.toFixed(3) }}</span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const fileInput = ref(null)
const selectedFile = ref(null)
const previewUrl = ref(null)
const results = ref([])
const isLoading = ref(false)
const errorMessage = ref(null)

const onFileSelected = (event) => {
  const file = event.target.files[0]
  if (!file) return
  
  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
  results.value = [] // Reset kết quả cũ khi chọn ảnh mới
  errorMessage.value = null
}

const searchImage = async () => {
  if (!selectedFile.value) return
  
  isLoading.value = true
  errorMessage.value = null
  
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  formData.append('top_k', 5) 

  try {
    // Gọi sang cổng 8000 của FastAPI Backend
    const response = await fetch('http://127.0.0.1:8000/search', {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Lỗi kết nối máy chủ!')
    }
    
    const data = await response.json()
    results.value = data.results
  } catch (error) {
    // Bắt lỗi "Failed to fetch" nếu chưa bật Backend
    if (error.message.includes('Failed to fetch')) {
      errorMessage.value = "Không thể kết nối đến AI Server. Bạn đã chạy lệnh 'python app.py' chưa?"
    } else {
      errorMessage.value = error.message
    }
  } finally {
    isLoading.value = false
  }
}
</script>

<style>
.animate-fade-in {
  animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
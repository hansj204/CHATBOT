// MBTI 데이터
const mbtiData = {
  e: 49,
  i: 51,
  s: 49,
  n: 51
};

const config = {
  type: 'bar',
  data: {
    labels: [''],
    datasets: [
      {
        label: 'E',
        data: [mbtiData.e],
        backgroundColor: 'rgba(75, 192, 192, 0.5)'
      }, {
        label: 'I',
        data: [mbtiData.i],
        backgroundColor: 'rgba(255, 99, 132, 0.5)'
      },
    ]
  },
  options: {
      indexAxis: 'y',
      scales: {
          x: {
              stacked: true,
              ticks: {
                display: false
              }
          },
          y: {
              stacked: true,
              max: 100
          }
      }
  }
}

// 캔버스 요소 가져오기
const ctx1 = document.getElementById('mbtiChart1').getContext('2d');
const ctx2 = document.getElementById('mbtiChart2').getContext('2d');
const ctx3 = document.getElementById('mbtiChart3').getContext('2d');
const ctx4 = document.getElementById('mbtiChart4').getContext('2d');

// 차트 생성
const mbtiChart1 = new Chart(ctx1, config);
const mbtiChart2 = new Chart(ctx2, config);
const mbtiChart3 = new Chart(ctx3, config);
const mbtiChart4 = new Chart(ctx4, config);

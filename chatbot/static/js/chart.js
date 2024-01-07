const createMbtiConfig = (id, label, data) => {
  const [label1, label2] = label;
  
  return new Chart(document.getElementById(id).getContext('2d'), {
    type: 'bar',
    data: {
      labels: [''],
      datasets: [
        {
          label: label1,
          data: [data[label1]],
          backgroundColor: 'rgba(245, 075, 015)'
        },
        {
          label: label2,
          data: [data[label2]],
          backgroundColor: 'rgba(242, 153, 015)'
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
  })
};

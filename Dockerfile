FROM python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir c:\home\kyc_face_chart
COPY kyc_face_chart.py /home/kyc_face_chart/kyc_face_chart.py
CMD python /home/kyc_face_chart/kyc_face_chart.py
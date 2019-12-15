from django.db import models


class UnprocessedData(models.Model):
    Age = models.IntegerField()
    Attrition = models.CharField(max_length=200)
    BusinessTravel = models.CharField(max_length=200)
    DailyRate = models.IntegerField()
    Department = models.CharField(max_length=200)
    DistanceFromHome = models.IntegerField()
    Education = models.IntegerField()
    EducationField = models.CharField(max_length=200)
    EmployeeCount = models.IntegerField()
    EmployeeNumber = models.IntegerField()
    EnvironmentSatisfaction = models.IntegerField()
    Gender = models.CharField(max_length=200)
    HourlyRate = models.IntegerField()
    JobInvolvement = models.IntegerField()
    JobLevel = models.IntegerField()
    JobRole = models.CharField(max_length=200)
    JobSatisfaction = models.IntegerField()
    MaritalStatus = models.CharField(max_length=200)
    MonthlyIncome = models.IntegerField()
    MonthlyRate = models.IntegerField()
    NumCompaniesWorked = models.IntegerField()
    Over18 = models.CharField(max_length=200)
    OverTime = models.CharField(max_length=200)
    PercentSalaryHike = models.IntegerField()
    PerformanceRating = models.IntegerField()
    RelationshipSatisfaction = models.IntegerField()
    StandardHours = models.IntegerField()
    StockOptionLevel = models.IntegerField()
    TotalWorkingYears = models.IntegerField()
    TrainingTimesLastYear = models.IntegerField()
    WorkLifeBalance = models.IntegerField()
    YearsAtCompany = models.IntegerField()
    YearsInCurrentRole = models.IntegerField()
    YearsSinceLastPromotion = models.IntegerField()
    YearsWithCurrManager = models.IntegerField()


class ModelDetail(models.Model):
    AlgorithmName = models.CharField(max_length=200)
    Accuracy = models.IntegerField()
    CorelationMatris = models.CharField(max_length=200)
    Date = models.DateTimeField(editable=False)


class PredictFormData(models.Model):
    algortihm=models.CharField(max_length=200)
    fname=models.CharField(max_length=200)


"""class Item(models.Model):
	todolist = models.ForeignKey(ToDoList, on_delete=models.CASCADE)
	text = models.CharField(max_length=300)
	complete = models.BooleanField()

	def __str__(self):
		return self.text"""

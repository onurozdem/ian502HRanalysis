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


class PredictFormData(models.Model):
    Algorithm = models.CharField(max_length=200)
    Age = models.IntegerField()
    DailyRate = models.IntegerField()
    DistanceFromHome = models.IntegerField()
    EducationField = models.CharField(max_length=200)
    EnvironmentSatisfaction = models.IntegerField()
    HourlyRate = models.IntegerField()
    JobInvolvement = models.IntegerField()
    JobLevel = models.IntegerField()
    JobRole = models.CharField(max_length=200)
    JobSatisfaction = models.IntegerField()
    MonthlyIncome = models.IntegerField()
    MonthlyRate = models.IntegerField()
    NumCompaniesWorked = models.IntegerField()
    OverTime = models.CharField(max_length=200)
    PercentSalaryHike = models.IntegerField()
    RelationshipSatisfaction = models.IntegerField()
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
    ModelScoreDict = models.CharField(max_length=500)
    Date = models.DateTimeField(editable=False, auto_now_add=True)


class ModelDetailFormData(models.Model):
    algortihm = models.CharField(max_length=200)


class ModelCompareFormData(models.Model):
    algortihm1 = models.CharField(max_length=200)
    algortihm2 = models.CharField(max_length=200)


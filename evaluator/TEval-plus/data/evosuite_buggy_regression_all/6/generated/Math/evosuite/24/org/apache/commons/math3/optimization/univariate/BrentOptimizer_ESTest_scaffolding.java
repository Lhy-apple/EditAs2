/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Wed Sep 27 00:49:46 GMT 2023
 */

package org.apache.commons.math3.optimization.univariate;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class BrentOptimizer_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "org.apache.commons.math3.optimization.univariate.BrentOptimizer"; 
    org.evosuite.runtime.GuiSupport.initialize(); 
    org.evosuite.runtime.RuntimeSettings.maxNumberOfThreads = 100; 
    org.evosuite.runtime.RuntimeSettings.maxNumberOfIterationsPerLoop = 10000; 
    org.evosuite.runtime.RuntimeSettings.mockSystemIn = true; 
    org.evosuite.runtime.RuntimeSettings.sandboxMode = org.evosuite.runtime.sandbox.Sandbox.SandboxMode.RECOMMENDED; 
    org.evosuite.runtime.sandbox.Sandbox.initializeSecurityManagerForSUT(); 
    org.evosuite.runtime.classhandling.JDKClassResetter.init();
    setSystemProperties();
    initializeClasses();
    org.evosuite.runtime.Runtime.getInstance().resetRuntime(); 
  } 

  @AfterClass 
  public static void clearEvoSuiteFramework(){ 
    Sandbox.resetDefaultSecurityManager(); 
    java.lang.System.setProperties((java.util.Properties) defaultProperties.clone()); 
  } 

  @Before 
  public void initTestCase(){ 
    threadStopper.storeCurrentThreads();
    threadStopper.startRecordingTime();
    org.evosuite.runtime.jvm.ShutdownHookHandler.getInstance().initHandler(); 
    org.evosuite.runtime.sandbox.Sandbox.goingToExecuteSUTCode(); 
    setSystemProperties(); 
    org.evosuite.runtime.GuiSupport.setHeadless(); 
    org.evosuite.runtime.Runtime.getInstance().resetRuntime(); 
    org.evosuite.runtime.agent.InstrumentingAgent.activate(); 
  } 

  @After 
  public void doneWithTestCase(){ 
    threadStopper.killAndJoinClientThreads();
    org.evosuite.runtime.jvm.ShutdownHookHandler.getInstance().safeExecuteAddedHooks(); 
    org.evosuite.runtime.classhandling.JDKClassResetter.reset(); 
    resetClasses(); 
    org.evosuite.runtime.sandbox.Sandbox.doneWithExecutingSUTCode(); 
    org.evosuite.runtime.agent.InstrumentingAgent.deactivate(); 
    org.evosuite.runtime.GuiSupport.restoreHeadlessMode(); 
  } 

  public static void setSystemProperties() {
 
    java.lang.System.setProperties((java.util.Properties) defaultProperties.clone()); 
    java.lang.System.setProperty("user.dir", "/data/lhy/TEval-plus"); 
    java.lang.System.setProperty("java.io.tmpdir", "/tmp"); 
  }

  private static void initializeClasses() {
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(BrentOptimizer_ESTest_scaffolding.class.getClassLoader() ,
      "org.apache.commons.math3.util.Precision",
      "org.apache.commons.math3.exception.util.ExceptionContextProvider",
      "org.apache.commons.math3.analysis.function.Tan",
      "org.apache.commons.math3.optimization.BaseOptimizer",
      "org.apache.commons.math3.optimization.univariate.BrentOptimizer",
      "org.apache.commons.math3.analysis.function.Asinh",
      "org.apache.commons.math3.analysis.DifferentiableUnivariateFunction",
      "org.apache.commons.math3.util.Incrementor$1",
      "org.apache.commons.math3.optimization.GoalType",
      "org.apache.commons.math3.exception.util.ArgUtils",
      "org.apache.commons.math3.analysis.differentiation.DerivativeStructure",
      "org.apache.commons.math3.exception.MathArithmeticException",
      "org.apache.commons.math3.optimization.univariate.UnivariatePointValuePair",
      "org.apache.commons.math3.exception.NumberIsTooSmallException",
      "org.apache.commons.math3.util.FastMath$ExpIntTable",
      "org.apache.commons.math3.util.FastMath$lnMant",
      "org.apache.commons.math3.exception.MathIllegalStateException",
      "org.apache.commons.math3.util.FastMath$ExpFracTable",
      "org.apache.commons.math3.analysis.differentiation.UnivariateDifferentiable",
      "org.apache.commons.math3.analysis.function.Expm1",
      "org.apache.commons.math3.exception.MathIllegalArgumentException",
      "org.apache.commons.math3.util.Incrementor",
      "org.apache.commons.math3.util.FastMath$CodyWaite",
      "org.apache.commons.math3.exception.MathIllegalNumberException",
      "org.apache.commons.math3.exception.util.LocalizedFormats",
      "org.apache.commons.math3.optimization.univariate.UnivariateOptimizer",
      "org.apache.commons.math3.util.Incrementor$MaxCountExceededCallback",
      "org.apache.commons.math3.optimization.univariate.BaseUnivariateOptimizer",
      "org.apache.commons.math3.util.FastMath",
      "org.apache.commons.math3.optimization.univariate.BaseAbstractUnivariateOptimizer",
      "org.apache.commons.math3.exception.MaxCountExceededException",
      "org.apache.commons.math3.optimization.ConvergenceChecker",
      "org.apache.commons.math3.FieldElement",
      "org.apache.commons.math3.exception.util.Localizable",
      "org.apache.commons.math3.exception.NotStrictlyPositiveException",
      "org.apache.commons.math3.exception.util.ExceptionContext",
      "org.apache.commons.math3.exception.NullArgumentException",
      "org.apache.commons.math3.exception.TooManyEvaluationsException",
      "org.apache.commons.math3.analysis.UnivariateFunction",
      "org.apache.commons.math3.analysis.function.HarmonicOscillator",
      "org.apache.commons.math3.util.FastMathLiteralArrays"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(BrentOptimizer_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "org.apache.commons.math3.optimization.univariate.BaseAbstractUnivariateOptimizer",
      "org.apache.commons.math3.util.FastMath",
      "org.apache.commons.math3.optimization.univariate.BrentOptimizer",
      "org.apache.commons.math3.optimization.GoalType",
      "org.apache.commons.math3.util.Incrementor",
      "org.apache.commons.math3.util.Incrementor$1",
      "org.apache.commons.math3.exception.MathIllegalArgumentException",
      "org.apache.commons.math3.exception.MathIllegalNumberException",
      "org.apache.commons.math3.exception.NumberIsTooSmallException",
      "org.apache.commons.math3.exception.util.LocalizedFormats",
      "org.apache.commons.math3.exception.util.ExceptionContext",
      "org.apache.commons.math3.exception.util.ArgUtils",
      "org.apache.commons.math3.exception.NotStrictlyPositiveException",
      "org.apache.commons.math3.analysis.function.Sin",
      "org.apache.commons.math3.analysis.function.Log1p",
      "org.apache.commons.math3.analysis.function.Inverse",
      "org.apache.commons.math3.analysis.differentiation.DerivativeStructure",
      "org.apache.commons.math3.analysis.differentiation.DSCompiler",
      "org.apache.commons.math3.analysis.function.Logit",
      "org.apache.commons.math3.analysis.function.Sqrt",
      "org.apache.commons.math3.exception.MathIllegalStateException",
      "org.apache.commons.math3.exception.MaxCountExceededException",
      "org.apache.commons.math3.exception.TooManyEvaluationsException",
      "org.apache.commons.math3.analysis.function.Ulp",
      "org.apache.commons.math3.analysis.function.Ceil",
      "org.apache.commons.math3.analysis.function.Cbrt",
      "org.apache.commons.math3.analysis.function.Gaussian",
      "org.apache.commons.math3.analysis.function.Exp",
      "org.apache.commons.math3.analysis.function.Sinh",
      "org.apache.commons.math3.optimization.univariate.UnivariatePointValuePair",
      "org.apache.commons.math3.analysis.function.Logistic",
      "org.apache.commons.math3.exception.NullArgumentException",
      "org.apache.commons.math3.analysis.function.Rint",
      "org.apache.commons.math3.analysis.function.Sigmoid",
      "org.apache.commons.math3.analysis.function.Asinh",
      "org.apache.commons.math3.util.FastMathLiteralArrays",
      "org.apache.commons.math3.util.FastMath$lnMant",
      "org.apache.commons.math3.analysis.function.Atanh",
      "org.apache.commons.math3.util.Precision",
      "org.apache.commons.math3.analysis.function.HarmonicOscillator",
      "org.apache.commons.math3.analysis.function.Abs",
      "org.apache.commons.math3.analysis.function.Power",
      "org.apache.commons.math3.analysis.function.Log10",
      "org.apache.commons.math3.analysis.function.Floor",
      "org.apache.commons.math3.analysis.function.Acosh",
      "org.apache.commons.math3.analysis.function.Constant",
      "org.apache.commons.math3.analysis.polynomials.PolynomialFunction",
      "org.apache.commons.math3.util.MathUtils",
      "org.apache.commons.math3.exception.NoDataException",
      "org.apache.commons.math3.analysis.function.Sinc",
      "org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction",
      "org.apache.commons.math3.analysis.function.Cos",
      "org.apache.commons.math3.analysis.function.Atan",
      "org.apache.commons.math3.analysis.function.Cosh",
      "org.apache.commons.math3.analysis.function.Acos",
      "org.apache.commons.math3.util.FastMath$CodyWaite",
      "org.apache.commons.math3.exception.NumberIsTooLargeException",
      "org.apache.commons.math3.exception.OutOfRangeException",
      "org.apache.commons.math3.analysis.function.Tan",
      "org.apache.commons.math3.analysis.function.Asin",
      "org.apache.commons.math3.analysis.function.Identity",
      "org.apache.commons.math3.analysis.function.Minus",
      "org.apache.commons.math3.analysis.polynomials.PolynomialFunctionNewtonForm",
      "org.apache.commons.math3.exception.DimensionMismatchException",
      "org.apache.commons.math3.analysis.polynomials.PolynomialFunctionLagrangeForm",
      "org.apache.commons.math3.util.MathArrays$OrderDirection",
      "org.apache.commons.math3.util.MathArrays",
      "org.apache.commons.math3.util.MathArrays$2",
      "org.apache.commons.math3.util.Pair",
      "org.apache.commons.math3.util.MathArrays$1",
      "org.apache.commons.math3.exception.NonMonotonicSequenceException",
      "org.apache.commons.math3.util.FastMath$ExpIntTable",
      "org.apache.commons.math3.util.FastMath$ExpFracTable",
      "org.apache.commons.math3.analysis.function.Tanh",
      "org.apache.commons.math3.analysis.function.Expm1",
      "org.apache.commons.math3.analysis.function.Signum",
      "org.apache.commons.math3.analysis.function.Log",
      "org.apache.commons.math3.analysis.function.StepFunction"
    );
  }
}
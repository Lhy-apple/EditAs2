/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Tue Sep 26 14:06:06 GMT 2023
 */

package org.apache.commons.math.analysis.solvers;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class BaseSecantSolver_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "org.apache.commons.math.analysis.solvers.BaseSecantSolver"; 
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
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(BaseSecantSolver_ESTest_scaffolding.class.getClassLoader() ,
      "org.apache.commons.math.exception.MathIllegalStateException",
      "org.apache.commons.math.util.Incrementor",
      "org.apache.commons.math.exception.NumberIsTooSmallException",
      "org.apache.commons.math.analysis.function.Inverse",
      "org.apache.commons.math.exception.NullArgumentException",
      "org.apache.commons.math.exception.util.ExceptionContext",
      "org.apache.commons.math.analysis.solvers.BaseSecantSolver",
      "org.apache.commons.math.analysis.solvers.UnivariateRealSolverUtils",
      "org.apache.commons.math.exception.NonMonotonousSequenceException",
      "org.apache.commons.math.util.FastMath",
      "org.apache.commons.math.util.MathUtils",
      "org.apache.commons.math.analysis.solvers.BaseSecantSolver$Method",
      "org.apache.commons.math.analysis.function.Sigmoid",
      "org.apache.commons.math.analysis.solvers.AllowedSolution",
      "org.apache.commons.math.analysis.solvers.UnivariateRealSolver",
      "org.apache.commons.math.exception.OutOfRangeException",
      "org.apache.commons.math.exception.NotStrictlyPositiveException",
      "org.apache.commons.math.analysis.function.Logit$1",
      "org.apache.commons.math.exception.NumberIsTooLargeException",
      "org.apache.commons.math.analysis.solvers.IllinoisSolver",
      "org.apache.commons.math.exception.NotFiniteNumberException",
      "org.apache.commons.math.analysis.DifferentiableUnivariateRealFunction",
      "org.apache.commons.math.exception.MathInternalError",
      "org.apache.commons.math.analysis.UnivariateRealFunction",
      "org.apache.commons.math.analysis.solvers.BaseUnivariateRealSolver",
      "org.apache.commons.math.exception.TooManyEvaluationsException",
      "org.apache.commons.math.analysis.solvers.BaseSecantSolver$1",
      "org.apache.commons.math.analysis.function.Logit",
      "org.apache.commons.math.exception.NotPositiveException",
      "org.apache.commons.math.analysis.solvers.RegulaFalsiSolver",
      "org.apache.commons.math.exception.util.Localizable",
      "org.apache.commons.math.exception.MathIllegalArgumentException",
      "org.apache.commons.math.analysis.solvers.AbstractUnivariateRealSolver",
      "org.apache.commons.math.exception.MaxCountExceededException",
      "org.apache.commons.math.exception.MathUserException",
      "org.apache.commons.math.analysis.solvers.BaseAbstractUnivariateRealSolver",
      "org.apache.commons.math.exception.MathArithmeticException",
      "org.apache.commons.math.analysis.function.Expm1",
      "org.apache.commons.math.analysis.function.Log10",
      "org.apache.commons.math.exception.DimensionMismatchException",
      "org.apache.commons.math.exception.util.LocalizedFormats",
      "org.apache.commons.math.analysis.solvers.PegasusSolver",
      "org.apache.commons.math.exception.MathIllegalNumberException",
      "org.apache.commons.math.analysis.solvers.BracketedUnivariateRealSolver",
      "org.apache.commons.math.exception.util.ExceptionContextProvider",
      "org.apache.commons.math.exception.NoBracketingException",
      "org.apache.commons.math.exception.util.ArgUtils"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(BaseSecantSolver_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "org.apache.commons.math.analysis.solvers.BaseAbstractUnivariateRealSolver",
      "org.apache.commons.math.analysis.solvers.AbstractUnivariateRealSolver",
      "org.apache.commons.math.analysis.solvers.BaseSecantSolver",
      "org.apache.commons.math.analysis.solvers.BaseSecantSolver$Method",
      "org.apache.commons.math.analysis.solvers.AllowedSolution",
      "org.apache.commons.math.analysis.solvers.BaseSecantSolver$1",
      "org.apache.commons.math.util.FastMath",
      "org.apache.commons.math.analysis.solvers.IllinoisSolver",
      "org.apache.commons.math.util.Incrementor",
      "org.apache.commons.math.exception.MathIllegalStateException",
      "org.apache.commons.math.exception.MaxCountExceededException",
      "org.apache.commons.math.exception.util.LocalizedFormats",
      "org.apache.commons.math.exception.util.ExceptionContext",
      "org.apache.commons.math.exception.util.ArgUtils",
      "org.apache.commons.math.exception.TooManyEvaluationsException",
      "org.apache.commons.math.analysis.solvers.RegulaFalsiSolver",
      "org.apache.commons.math.analysis.function.Asin",
      "org.apache.commons.math.util.MathUtils",
      "org.apache.commons.math.analysis.solvers.UnivariateRealSolverUtils",
      "org.apache.commons.math.exception.MathIllegalArgumentException",
      "org.apache.commons.math.exception.NullArgumentException",
      "org.apache.commons.math.analysis.solvers.PegasusSolver",
      "org.apache.commons.math.analysis.function.Expm1",
      "org.apache.commons.math.analysis.function.Acosh",
      "org.apache.commons.math.exception.MathIllegalNumberException",
      "org.apache.commons.math.exception.NumberIsTooLargeException",
      "org.apache.commons.math.analysis.polynomials.PolynomialFunctionLagrangeForm",
      "org.apache.commons.math.exception.DimensionMismatchException",
      "org.apache.commons.math.analysis.polynomials.PolynomialFunction",
      "org.apache.commons.math.analysis.polynomials.PolynomialSplineFunction",
      "org.apache.commons.math.analysis.function.Sigmoid",
      "org.apache.commons.math.analysis.function.Sigmoid$1",
      "org.apache.commons.math.analysis.function.Acos",
      "org.apache.commons.math.analysis.function.Cbrt",
      "org.apache.commons.math.exception.NoBracketingException",
      "org.apache.commons.math.analysis.function.Sinh",
      "org.apache.commons.math.analysis.function.Gaussian",
      "org.apache.commons.math.exception.NumberIsTooSmallException",
      "org.apache.commons.math.exception.NotStrictlyPositiveException",
      "org.apache.commons.math.analysis.polynomials.PolynomialFunctionNewtonForm",
      "org.apache.commons.math.exception.NoDataException",
      "org.apache.commons.math.analysis.function.Cos",
      "org.apache.commons.math.analysis.function.HarmonicOscillator",
      "org.apache.commons.math.analysis.function.HarmonicOscillator$1",
      "org.apache.commons.math.analysis.function.Tan",
      "org.apache.commons.math.analysis.function.Logistic",
      "org.apache.commons.math.analysis.function.Identity",
      "org.apache.commons.math.analysis.function.Constant",
      "org.apache.commons.math.analysis.function.Rint",
      "org.apache.commons.math.analysis.function.Log",
      "org.apache.commons.math.analysis.function.Atanh",
      "org.apache.commons.math.analysis.function.Minus",
      "org.apache.commons.math.analysis.function.Inverse",
      "org.apache.commons.math.analysis.function.Power",
      "org.apache.commons.math.analysis.function.Log10",
      "org.apache.commons.math.analysis.function.Sqrt",
      "org.apache.commons.math.analysis.function.Abs",
      "org.apache.commons.math.analysis.function.Log1p",
      "org.apache.commons.math.analysis.function.Asinh",
      "org.apache.commons.math.analysis.function.Sinc",
      "org.apache.commons.math.analysis.function.StepFunction",
      "org.apache.commons.math.util.MathUtils$OrderDirection",
      "org.apache.commons.math.util.MathUtils$2",
      "org.apache.commons.math.exception.NonMonotonousSequenceException",
      "org.apache.commons.math.analysis.function.Logistic$1",
      "org.apache.commons.math.analysis.function.Logit",
      "org.apache.commons.math.analysis.function.Logit$1",
      "org.apache.commons.math.analysis.function.Signum",
      "org.apache.commons.math.exception.OutOfRangeException",
      "org.apache.commons.math.analysis.function.Ceil",
      "org.apache.commons.math.analysis.function.Ulp",
      "org.apache.commons.math.analysis.function.Floor",
      "org.apache.commons.math.analysis.function.Tanh",
      "org.apache.commons.math.analysis.function.Exp",
      "org.apache.commons.math.util.Pair",
      "org.apache.commons.math.util.MathUtils$1",
      "org.apache.commons.math.analysis.function.Sin",
      "org.apache.commons.math.analysis.function.Atan",
      "org.apache.commons.math.analysis.function.Gaussian$1",
      "org.apache.commons.math.analysis.function.Cosh"
    );
  }
}

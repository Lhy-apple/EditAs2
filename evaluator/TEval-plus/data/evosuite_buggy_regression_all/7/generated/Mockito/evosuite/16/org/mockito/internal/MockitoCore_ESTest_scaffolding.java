/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Sat Jul 29 20:07:22 GMT 2023
 */

package org.mockito.internal;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class MockitoCore_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "org.mockito.internal.MockitoCore"; 
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
    java.lang.System.setProperty("user.dir", "/data/swf/zenodo_replication_package_new"); 
    java.lang.System.setProperty("java.io.tmpdir", "/tmp"); 
  }

  private static void initializeClasses() {
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(MockitoCore_ESTest_scaffolding.class.getClassLoader() ,
      "org.mockito.cglib.core.MethodInfo",
      "org.mockito.cglib.proxy.Callback",
      "org.mockito.exceptions.misusing.UnfinishedVerificationException",
      "org.mockito.cglib.proxy.FixedValueGenerator",
      "org.mockito.cglib.proxy.MethodInterceptorGenerator$1",
      "org.mockito.cglib.proxy.InvocationHandlerGenerator",
      "org.mockito.internal.stubbing.InvocationContainer",
      "org.mockito.internal.util.ObjectMethodsGuru",
      "org.mockito.internal.creation.MockSettingsImpl",
      "org.mockito.internal.creation.jmock.ClassImposterizer$ClassWithSuperclassToWorkAroundCglibBug",
      "org.mockito.cglib.core.LocalVariablesSorter",
      "org.mockito.cglib.core.ClassNameReader$1",
      "org.mockito.internal.creation.jmock.ClassImposterizer$3",
      "org.mockito.cglib.core.ReflectUtils",
      "org.mockito.internal.stubbing.InvocationContainerImpl",
      "org.mockito.cglib.proxy.CallbackGenerator",
      "org.mockito.cglib.core.ClassInfo",
      "org.mockito.cglib.proxy.FixedValue",
      "org.mockito.cglib.core.ObjectSwitchCallback",
      "org.mockito.exceptions.misusing.NotAMockException",
      "org.mockito.internal.stubbing.OngoingStubbingImpl",
      "org.mockito.cglib.core.ClassEmitter$FieldInfo",
      "org.mockito.internal.invocation.realmethod.RealMethod",
      "org.mockito.internal.creation.jmock.ClassImposterizer",
      "org.mockito.internal.creation.cglib.CGLIBHacker",
      "org.mockito.cglib.core.GeneratorStrategy",
      "org.mockito.internal.InOrderImpl",
      "org.objenesis.ObjenesisStd",
      "org.mockito.internal.progress.MockingProgress",
      "org.mockito.internal.util.MockitoLogger",
      "org.mockito.exceptions.misusing.MissingMethodInvocationException",
      "org.mockito.exceptions.verification.SmartNullPointerException",
      "org.mockito.exceptions.verification.TooLittleActualInvocations",
      "org.mockito.internal.progress.MockingProgressImpl",
      "org.mockito.cglib.core.Local",
      "org.mockito.cglib.core.ClassNameReader$EarlyExitException",
      "org.mockito.internal.configuration.GlobalConfiguration",
      "org.objenesis.Objenesis",
      "org.mockito.stubbing.DeprecatedOngoingStubbing",
      "org.mockito.cglib.proxy.MethodInterceptor",
      "org.mockito.exceptions.verification.TooManyActualInvocations",
      "org.mockito.cglib.proxy.CallbackGenerator$Context",
      "org.mockito.asm.Item",
      "org.mockito.asm.FieldVisitor",
      "org.mockito.cglib.core.ClassEmitter",
      "org.mockito.cglib.core.Transformer",
      "org.objenesis.strategy.InstantiatorStrategy",
      "org.mockito.exceptions.misusing.MockitoConfigurationException",
      "org.mockito.cglib.core.AbstractClassGenerator",
      "org.mockito.internal.progress.IOngoingStubbing",
      "org.mockito.internal.invocation.MockitoMethod",
      "org.mockito.internal.debugging.Localized",
      "org.mockito.cglib.core.CodeEmitter$State",
      "org.mockito.internal.MockitoCore",
      "org.mockito.internal.util.ListUtil$Filter",
      "org.mockito.cglib.core.KeyFactory$Generator",
      "org.mockito.stubbing.Stubber",
      "org.mockito.asm.Type",
      "org.mockito.stubbing.Answer",
      "org.mockito.internal.exceptions.base.StackTraceFilter",
      "org.mockito.cglib.core.ClassEmitter$3",
      "org.mockito.invocation.InvocationOnMock",
      "org.mockito.internal.progress.ArgumentMatcherStorageImpl",
      "org.mockito.cglib.proxy.Enhancer",
      "org.mockito.cglib.core.ProcessArrayCallback",
      "org.mockito.internal.verification.NoMoreInteractions",
      "org.mockito.asm.Opcodes",
      "org.objenesis.instantiator.ObjectInstantiator",
      "org.objenesis.strategy.StdInstantiatorStrategy",
      "org.mockito.cglib.core.ClassEmitter$1",
      "org.mockito.internal.creation.MethodInterceptorFilter",
      "org.mockito.configuration.IMockitoConfiguration",
      "org.mockito.exceptions.misusing.WrongTypeOfReturnValue",
      "org.mockito.MockSettings",
      "org.mockito.stubbing.OngoingStubbing",
      "org.mockito.cglib.core.Predicate",
      "org.mockito.internal.progress.ArgumentMatcherStorage",
      "org.mockito.cglib.proxy.ProxyRefDispatcher",
      "org.mockito.cglib.core.EmitUtils$ArrayDelimiters",
      "org.mockito.exceptions.base.MockitoAssertionError",
      "org.mockito.internal.creation.jmock.ClassImposterizer$1",
      "org.mockito.internal.creation.jmock.ClassImposterizer$2",
      "org.mockito.cglib.proxy.LazyLoaderGenerator",
      "org.mockito.internal.invocation.InvocationMarker",
      "org.mockito.asm.ClassVisitor",
      "org.mockito.exceptions.verification.NeverWantedButInvoked",
      "org.mockito.internal.util.CreationValidator",
      "org.mockito.cglib.core.CodeGenerationException",
      "org.mockito.cglib.core.CollectionUtils",
      "org.hamcrest.Matcher",
      "org.mockito.asm.MethodAdapter",
      "org.mockito.internal.util.MockName",
      "org.mockito.cglib.core.KeyFactory$2",
      "org.mockito.cglib.core.KeyFactory$1",
      "org.objenesis.strategy.BaseInstantiatorStrategy",
      "org.mockito.cglib.core.Customizer",
      "org.mockito.configuration.AnnotationEngine",
      "org.mockito.cglib.core.EmitUtils",
      "org.mockito.internal.invocation.MatchersBinder",
      "org.mockito.cglib.core.Constants",
      "org.mockito.exceptions.Reporter",
      "org.mockito.exceptions.verification.VerificationInOrderFailure",
      "org.mockito.configuration.DefaultMockitoConfiguration",
      "org.mockito.cglib.proxy.LazyLoader",
      "org.mockito.exceptions.misusing.NullInsteadOfMockException",
      "org.mockito.stubbing.VoidMethodStubbable",
      "org.mockito.cglib.core.DebuggingClassWriter",
      "org.mockito.internal.util.StringJoiner",
      "org.mockito.cglib.core.NamingPolicy",
      "org.mockito.internal.creation.jmock.SearchingClassLoader",
      "org.mockito.cglib.proxy.NoOp",
      "org.mockito.internal.verification.RegisteredInvocations",
      "org.mockito.cglib.core.LocalVariablesSorter$State",
      "org.mockito.internal.MockHandler",
      "org.mockito.cglib.proxy.InvocationHandler",
      "org.mockito.cglib.core.ReflectUtils$4",
      "org.mockito.cglib.core.ReflectUtils$2",
      "org.mockito.cglib.core.ReflectUtils$3",
      "org.mockito.asm.ByteVector",
      "org.mockito.cglib.core.ReflectUtils$1",
      "org.mockito.cglib.core.DebuggingClassWriter$1",
      "org.mockito.internal.creation.MockitoMethodProxy",
      "org.mockito.internal.reporting.PrintingFriendlyInvocation",
      "org.mockito.internal.creation.cglib.MockitoNamingPolicy",
      "org.mockito.exceptions.verification.ArgumentsAreDifferent",
      "org.mockito.cglib.core.AbstractClassGenerator$1",
      "org.mockito.cglib.core.DefaultGeneratorStrategy",
      "org.mockito.cglib.core.ProcessSwitchCallback",
      "org.mockito.cglib.core.ClassNameReader",
      "org.hamcrest.SelfDescribing",
      "org.mockito.cglib.core.AbstractClassGenerator$Source",
      "org.mockito.asm.FieldWriter",
      "org.mockito.exceptions.misusing.InvalidUseOfMatchersException",
      "org.mockito.cglib.proxy.MethodInterceptorGenerator",
      "org.mockito.internal.invocation.Invocation",
      "org.mockito.exceptions.misusing.UnfinishedStubbingException",
      "org.mockito.cglib.proxy.Dispatcher",
      "org.mockito.internal.debugging.DebuggingInfo",
      "org.mockito.cglib.core.EmitUtils$ParameterTyper",
      "org.mockito.cglib.core.DefaultNamingPolicy",
      "org.mockito.exceptions.verification.NoInteractionsWanted",
      "org.mockito.cglib.core.TypeUtils",
      "org.mockito.cglib.core.CodeEmitter",
      "org.mockito.internal.invocation.CapturesArgumensFromInvocation",
      "org.mockito.cglib.proxy.DispatcherGenerator",
      "org.mockito.exceptions.PrintableInvocation",
      "org.mockito.asm.ClassReader",
      "org.mockito.internal.configuration.ClassPathLoader",
      "org.mockito.exceptions.base.MockitoException",
      "org.mockito.internal.exceptions.base.ConditionalStackTraceFilter",
      "org.mockito.internal.verification.api.VerificationData",
      "org.mockito.internal.stubbing.StubberImpl",
      "org.mockito.internal.MockHandlerInterface",
      "org.mockito.internal.verification.Only",
      "org.mockito.exceptions.verification.WantedButNotInvoked",
      "org.mockito.internal.progress.ThreadSafeMockingProgress",
      "org.mockito.asm.MethodWriter",
      "org.mockito.internal.stubbing.ConsecutiveStubbing",
      "org.mockito.asm.Edge",
      "org.mockito.asm.Label",
      "org.mockito.internal.verification.api.VerificationMode",
      "org.mockito.internal.invocation.InvocationsFinder",
      "org.mockito.cglib.core.Signature",
      "org.mockito.internal.stubbing.BaseStubbing",
      "org.mockito.cglib.proxy.CallbackInfo",
      "org.mockito.internal.debugging.Location",
      "org.mockito.cglib.proxy.CallbackFilter",
      "org.mockito.asm.Attribute",
      "org.mockito.InOrder",
      "org.mockito.internal.MockitoInvocationHandler",
      "org.mockito.cglib.core.EmitUtils$8",
      "org.mockito.asm.AnnotationVisitor",
      "org.mockito.cglib.core.EmitUtils$9",
      "org.mockito.asm.ClassAdapter",
      "org.mockito.cglib.proxy.NoOpGenerator",
      "org.mockito.cglib.core.EmitUtils$7",
      "org.mockito.internal.util.MockUtil",
      "org.mockito.cglib.proxy.Enhancer$EnhancerKey",
      "org.mockito.cglib.proxy.Enhancer$1",
      "org.mockito.asm.MethodVisitor",
      "org.objenesis.ObjenesisBase",
      "org.mockito.asm.Frame",
      "org.mockito.asm.ClassWriter",
      "org.mockito.cglib.core.KeyFactory",
      "org.mockito.cglib.core.ClassGenerator"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(MockitoCore_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "org.mockito.internal.MockitoCore",
      "org.mockito.exceptions.Reporter",
      "org.mockito.internal.util.MockUtil",
      "org.mockito.internal.util.CreationValidator",
      "org.mockito.internal.progress.ThreadSafeMockingProgress",
      "org.mockito.internal.creation.MockSettingsImpl",
      "org.mockito.exceptions.base.MockitoException",
      "org.mockito.internal.util.StringJoiner",
      "org.mockito.internal.exceptions.base.ConditionalStackTraceFilter",
      "org.mockito.internal.configuration.GlobalConfiguration",
      "org.mockito.configuration.DefaultMockitoConfiguration",
      "org.mockito.internal.configuration.ClassPathLoader",
      "org.mockito.internal.exceptions.base.StackTraceFilter",
      "org.mockito.internal.stubbing.InvocationContainerImpl",
      "org.mockito.internal.verification.RegisteredInvocations",
      "org.mockito.internal.stubbing.BaseStubbing",
      "org.mockito.internal.stubbing.OngoingStubbingImpl",
      "org.mockito.internal.stubbing.answers.Returns",
      "org.mockito.exceptions.misusing.NullInsteadOfMockException",
      "org.mockito.internal.progress.MockingProgressImpl",
      "org.mockito.internal.progress.ArgumentMatcherStorageImpl",
      "org.mockito.internal.debugging.DebuggingInfo",
      "org.mockito.internal.debugging.Location",
      "org.mockito.exceptions.misusing.MissingMethodInvocationException",
      "org.mockito.cglib.core.AbstractClassGenerator",
      "org.mockito.cglib.proxy.Enhancer$1",
      "org.mockito.cglib.core.AbstractClassGenerator$Source",
      "org.mockito.cglib.core.CollectionUtils",
      "org.mockito.cglib.core.TypeUtils",
      "org.mockito.cglib.core.Signature",
      "org.mockito.asm.Type",
      "org.mockito.cglib.core.KeyFactory$1",
      "org.mockito.cglib.core.KeyFactory$2",
      "org.mockito.cglib.core.KeyFactory",
      "org.mockito.cglib.core.KeyFactory$Generator",
      "org.mockito.cglib.core.DefaultGeneratorStrategy",
      "org.mockito.cglib.core.DefaultNamingPolicy",
      "org.mockito.asm.ClassWriter",
      "org.mockito.cglib.core.DebuggingClassWriter",
      "org.mockito.asm.ByteVector",
      "org.mockito.asm.Item",
      "org.mockito.asm.ClassAdapter",
      "org.mockito.cglib.core.ClassEmitter",
      "org.mockito.cglib.core.ReflectUtils$1",
      "org.mockito.cglib.core.ReflectUtils$2",
      "org.mockito.cglib.core.ReflectUtils",
      "org.mockito.cglib.core.AbstractClassGenerator$1",
      "org.mockito.cglib.core.ClassInfo",
      "org.mockito.cglib.core.ClassEmitter$1",
      "org.mockito.cglib.core.EmitUtils$ArrayDelimiters",
      "org.mockito.cglib.core.EmitUtils",
      "org.mockito.asm.MethodWriter",
      "org.mockito.asm.Label",
      "org.mockito.cglib.core.Constants",
      "org.mockito.asm.MethodAdapter",
      "org.mockito.cglib.core.LocalVariablesSorter",
      "org.mockito.cglib.core.CodeEmitter",
      "org.mockito.cglib.core.LocalVariablesSorter$State",
      "org.mockito.cglib.core.MethodInfo",
      "org.mockito.cglib.core.CodeEmitter$State",
      "org.mockito.asm.Frame",
      "org.mockito.cglib.core.ClassEmitter$FieldInfo",
      "org.mockito.asm.FieldWriter",
      "org.mockito.asm.Edge",
      "org.mockito.cglib.core.EmitUtils$7",
      "org.mockito.cglib.core.Local",
      "org.mockito.cglib.core.EmitUtils$8",
      "org.mockito.cglib.core.EmitUtils$9",
      "org.mockito.cglib.core.DebuggingClassWriter$1",
      "org.mockito.asm.ClassReader",
      "org.mockito.cglib.core.ClassNameReader$EarlyExitException",
      "org.mockito.cglib.core.ClassNameReader",
      "org.mockito.cglib.core.ClassNameReader$1",
      "org.mockito.cglib.proxy.Enhancer",
      "org.mockito.exceptions.misusing.NotAMockException",
      "org.mockito.stubbing.ClonesArguments",
      "org.objenesis.ObjenesisBase",
      "org.objenesis.ObjenesisStd",
      "org.objenesis.strategy.BaseInstantiatorStrategy",
      "org.objenesis.strategy.StdInstantiatorStrategy",
      "org.mockito.internal.creation.cglib.MockitoNamingPolicy",
      "org.mockito.internal.creation.jmock.ClassImposterizer$1",
      "org.mockito.internal.creation.jmock.ClassImposterizer$2",
      "org.mockito.internal.creation.jmock.ClassImposterizer",
      "org.mockito.internal.util.MockName",
      "org.mockito.internal.MockHandler",
      "org.mockito.internal.invocation.MatchersBinder",
      "org.mockito.internal.creation.MethodInterceptorFilter",
      "org.mockito.internal.creation.cglib.CGLIBHacker",
      "org.mockito.internal.util.ObjectMethodsGuru",
      "org.mockito.internal.creation.jmock.ClassImposterizer$3",
      "org.mockito.internal.creation.jmock.SearchingClassLoader",
      "org.mockito.cglib.proxy.NoOpGenerator",
      "org.mockito.cglib.proxy.MethodInterceptorGenerator$1",
      "org.mockito.cglib.proxy.MethodInterceptorGenerator",
      "org.mockito.cglib.proxy.InvocationHandlerGenerator",
      "org.mockito.cglib.proxy.LazyLoaderGenerator",
      "org.mockito.cglib.proxy.DispatcherGenerator",
      "org.mockito.cglib.proxy.FixedValueGenerator",
      "org.mockito.cglib.proxy.CallbackInfo",
      "org.mockito.cglib.proxy.MethodProxy",
      "org.mockito.internal.verification.AtLeast",
      "org.mockito.internal.verification.Only",
      "org.mockito.internal.invocation.InvocationsFinder",
      "org.mockito.internal.invocation.InvocationMarker",
      "org.mockito.internal.stubbing.ConsecutiveStubbing",
      "org.mockito.internal.debugging.Localized",
      "org.mockito.internal.verification.NoMoreInteractions",
      "org.mockito.internal.verification.AtMost",
      "org.mockito.internal.verification.RegisteredInvocations$RemoveToString",
      "org.mockito.internal.util.ListUtil",
      "org.mockito.internal.stubbing.StubberImpl",
      "org.mockito.exceptions.misusing.UnfinishedStubbingException",
      "org.mockito.internal.verification.Times",
      "org.mockito.internal.stubbing.answers.ThrowsException",
      "org.mockito.internal.verification.InOrderWrapper",
      "org.mockito.exceptions.misusing.UnfinishedVerificationException",
      "org.mockito.internal.verification.VerificationDataImpl",
      "org.mockito.internal.verification.checkers.MissingInvocationInOrderChecker",
      "org.mockito.internal.verification.checkers.AtLeastXNumberOfInvocationsInOrderChecker",
      "org.mockito.internal.invocation.InvocationsFinder$RemoveNotMatching",
      "org.mockito.internal.invocation.InvocationsFinder$RemoveUnverifiedInOrder",
      "org.mockito.exceptions.base.MockitoAssertionError",
      "org.mockito.exceptions.verification.WantedButNotInvoked",
      "org.mockito.internal.invocation.SerializableMethod",
      "org.mockito.internal.invocation.InvocationMatcher",
      "org.mockito.cglib.proxy.MethodProxy$CreateInfo",
      "org.mockito.internal.creation.AbstractMockitoMethodProxy",
      "org.mockito.internal.creation.DelegatingMockitoMethodProxy",
      "org.mockito.internal.invocation.realmethod.CGLIBProxyRealMethod",
      "org.mockito.internal.stubbing.StubbedInvocationMatcher",
      "org.mockito.internal.creation.SerializableMockitoMethodProxy",
      "org.mockito.internal.util.reflection.Whitebox",
      "org.mockito.internal.invocation.realmethod.FilteredCGLIBProxyRealMethod",
      "org.mockito.internal.invocation.Invocation",
      "org.mockito.internal.stubbing.answers.CallsRealMethods",
      "org.mockito.internal.verification.checkers.MissingInvocationChecker",
      "org.mockito.cglib.proxy.MethodProxy$FastClassInfo",
      "org.mockito.cglib.reflect.FastClass$Generator",
      "org.mockito.cglib.reflect.FastClassEmitter",
      "org.mockito.cglib.core.VisibilityPredicate",
      "org.mockito.cglib.core.DuplicatesPredicate",
      "org.mockito.cglib.core.MethodWrapper",
      "org.mockito.cglib.reflect.FastClassEmitter$1",
      "org.mockito.cglib.reflect.FastClassEmitter$3",
      "org.mockito.cglib.core.EmitUtils$5",
      "org.mockito.cglib.core.EmitUtils$6",
      "org.mockito.cglib.core.MethodInfoTransformer",
      "org.mockito.cglib.core.ReflectUtils$3",
      "org.mockito.cglib.reflect.FastClassEmitter$GetIndexCallback",
      "org.mockito.cglib.core.EmitUtils$10",
      "org.mockito.cglib.core.EmitUtils$11",
      "org.mockito.cglib.core.EmitUtils$12",
      "org.mockito.cglib.core.EmitUtils$13",
      "org.mockito.cglib.core.EmitUtils$14",
      "org.mockito.cglib.core.EmitUtils$15",
      "org.mockito.cglib.core.EmitUtils$16",
      "org.mockito.cglib.core.Block",
      "org.mockito.cglib.reflect.FastClassEmitter$4",
      "org.mockito.cglib.core.ReflectUtils$4",
      "org.mockito.asm.Handler",
      "org.mockito.cglib.reflect.FastClass",
      "org.mockito.internal.invocation.AllInvocationsFinder",
      "org.mockito.internal.invocation.AllInvocationsFinder$SequenceNumberComparator",
      "org.mockito.internal.reporting.PrintSettings",
      "org.mockito.internal.verification.checkers.AtLeastXNumberOfInvocationsChecker",
      "org.mockito.exceptions.Discrepancy",
      "org.mockito.internal.verification.checkers.AtLeastDiscrepancy",
      "org.mockito.internal.InOrderImpl"
    );
  }
}

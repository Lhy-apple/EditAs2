/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:09:15 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.FunctionBuilder;
import com.google.javascript.rhino.jstype.FunctionParamBuilder;
import com.google.javascript.rhino.jstype.InstanceObjectType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NullType;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FunctionBuilder_ESTest extends FunctionBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      FunctionBuilder functionBuilder0 = new FunctionBuilder(jSTypeRegistry0);
      FunctionBuilder functionBuilder1 = functionBuilder0.withTemplateName("with");
      assertSame(functionBuilder0, functionBuilder1);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FunctionBuilder functionBuilder0 = new FunctionBuilder(jSTypeRegistry0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "q}0E:/elvS1~b");
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, errorFunctionType0);
      FunctionBuilder functionBuilder1 = functionBuilder0.withTypeOfThis(instanceObjectType0);
      assertSame(functionBuilder1, functionBuilder0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FunctionBuilder functionBuilder0 = new FunctionBuilder(jSTypeRegistry0);
      FunctionParamBuilder functionParamBuilder0 = new FunctionParamBuilder(jSTypeRegistry0);
      FunctionBuilder functionBuilder1 = functionBuilder0.withParams(functionParamBuilder0);
      assertSame(functionBuilder0, functionBuilder1);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      FunctionBuilder functionBuilder0 = new FunctionBuilder(jSTypeRegistry0);
      FunctionBuilder functionBuilder1 = functionBuilder0.forConstructor();
      assertSame(functionBuilder1, functionBuilder0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FunctionBuilder functionBuilder0 = new FunctionBuilder(jSTypeRegistry0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "q}0E:/elvS1~b");
      FunctionBuilder functionBuilder1 = functionBuilder0.copyFromOtherFunction(errorFunctionType0);
      assertSame(functionBuilder0, functionBuilder1);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FunctionBuilder functionBuilder0 = new FunctionBuilder(jSTypeRegistry0);
      NullType nullType0 = new NullType(jSTypeRegistry0);
      FunctionBuilder functionBuilder1 = functionBuilder0.withInferredReturnType(nullType0);
      assertSame(functionBuilder1, functionBuilder0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      FunctionBuilder functionBuilder0 = new FunctionBuilder(jSTypeRegistry0);
      Node node0 = Node.newString(".pu5e!EAYq[nKz");
      FunctionBuilder functionBuilder1 = functionBuilder0.withSourceNode(node0);
      assertSame(functionBuilder0, functionBuilder1);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FunctionBuilder functionBuilder0 = new FunctionBuilder(jSTypeRegistry0);
      FunctionBuilder functionBuilder1 = functionBuilder0.withName((String) null);
      assertSame(functionBuilder1, functionBuilder0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      FunctionBuilder functionBuilder0 = new FunctionBuilder(jSTypeRegistry0);
      FunctionBuilder functionBuilder1 = functionBuilder0.forNativeType();
      assertSame(functionBuilder1, functionBuilder0);
  }
}
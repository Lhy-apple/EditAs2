/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:28:40 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import com.google.common.base.Predicate;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NamedType;
import com.google.javascript.rhino.jstype.NoResolvedType;
import com.google.javascript.rhino.jstype.SimpleSlot;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NamedType_ESTest extends NamedType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, ">(S(VK", ">(S(VK", 2236, 2236);
      boolean boolean0 = namedType0.isNamedType();
      assertTrue(boolean0);
      assertEquals(">(S(VK", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Named type with empty name component", "Not declared as a constructor", 1, 0);
      String string0 = namedType0.toString();
      assertEquals("Named type with empty name component", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "-S1.A/.\"C4<7i&[rZv", "-S1.A/.\"C4<7i&[rZv", (-1212), (-1212));
      boolean boolean0 = namedType0.isNominalType();
      assertEquals("-S1.A/.\"C4<7i&[rZv", namedType0.getReferenceName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "SyntaxError", "Unknown class name", 0, 1);
      namedType0.defineDeclaredProperty("Not declared as a type name", noResolvedType0, (Node) null);
      namedType0.resolveInternal(simpleErrorReporter0, noResolvedType0);
      assertTrue(namedType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "com.google.javascript.rhino.jstype.NamedType", "com.google.javascript.rhino.jstype.NamedType", 28, (-919));
      namedType0.hashCode();
      assertEquals("com.google.javascript.rhino.jstype.NamedType", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "prototype", "Not declared as a constructor", 1, 1);
      namedType0.resolveInternal(simpleErrorReporter0, noResolvedType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "com.google.javascript.rhino.jstype.NamedType", "com.google.javascript.rhino.jstype.NamedType", 28, 28);
      Node node0 = Node.newString((-2426), "Named type with empty name component");
      namedType0.defineProperty("Unknown class name", (JSType) null, false, node0);
      boolean boolean0 = namedType0.defineProperty("com.google.javascript.rhino.jstype.NamedType", (JSType) null, false, node0);
      assertEquals("com.google.javascript.rhino.jstype.NamedType", namedType0.getReferenceName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Null", "Xtaq j3y+9zr#Uq", 1, 1);
      namedType0.resolveInternal(simpleErrorReporter0, noResolvedType0);
      assertEquals("Null", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      jSTypeRegistry0.setLastGeneration(false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "SyntaxError", "Not declared as a type name", 0, 0);
      namedType0.resolveInternal(simpleErrorReporter0, noResolvedType0);
      assertTrue(namedType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      jSTypeRegistry0.setLastGeneration(false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Not declared as a constructor", "Not declared as a type name", 1, 0);
      namedType0.resolveInternal(simpleErrorReporter0, noResolvedType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "prototype", "Unknown class name", 1, 0);
      JSTypeNative jSTypeNative0 = JSTypeNative.URI_ERROR_FUNCTION_TYPE;
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      namedType0.resolveInternal(simpleErrorReporter0, functionType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "SyntaxError", "Not declared as a type name", 0, 0);
      Predicate<JSType> predicate0 = (Predicate<JSType>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(predicate0).apply(any(com.google.javascript.rhino.jstype.JSType.class));
      namedType0.setValidator(predicate0);
      namedType0.resolveInternal(simpleErrorReporter0, noResolvedType0);
      assertTrue(namedType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      jSTypeRegistry0.forwardDeclareType("Not declared as a type name");
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Not declared as a type name", "Not declared as a constructor", 1, 1);
      namedType0.resolveInternal(simpleErrorReporter0, noResolvedType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "7", "7", (-131), (-131));
      SimpleSlot simpleSlot0 = new SimpleSlot("", (JSType) null, false);
      namedType0.getTypedefType(simpleErrorReporter0, simpleSlot0, "Not declared as a constructor");
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", (-1212), (-1212));
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Named type with empty name component");
      namedType0.resolveInternal(simpleErrorReporter0, errorFunctionType0);
      Predicate<JSType> predicate0 = (Predicate<JSType>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(predicate0).apply(any(com.google.javascript.rhino.jstype.JSType.class));
      boolean boolean0 = namedType0.setValidator(predicate0);
      assertTrue(namedType0.isResolved());
      assertFalse(boolean0);
  }
}
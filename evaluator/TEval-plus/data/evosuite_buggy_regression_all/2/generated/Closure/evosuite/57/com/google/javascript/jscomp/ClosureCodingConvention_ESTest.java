/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:30:14 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.CodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.GoogleCodingConvention;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.ObjectType;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ClosureCodingConvention_ESTest extends ClosureCodingConvention_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      // Undeclared exception!
      try { 
        closureCodingConvention0.extractClassNameIfRequire((Node) null, (Node) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Node node0 = new Node(38);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      boolean boolean0 = closureCodingConvention0.isVarArgsParameter(node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      String string0 = googleCodingConvention0.getExportSymbolFunction();
      assertEquals("goog.exportSymbol", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      Node node0 = new Node((-553), (-553), (-553));
      boolean boolean0 = closureCodingConvention0.isOptionalParameter(node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      boolean boolean0 = googleCodingConvention0.isSuperClassReference("rfP(#<cV\"=");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      String string0 = googleCodingConvention0.getAbstractMethodName();
      assertEquals("goog.abstractMethod", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      boolean boolean0 = closureCodingConvention0.isPrivate("");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      String string0 = closureCodingConvention0.getExportPropertyFunction();
      assertEquals("goog.exportProperty", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      // Undeclared exception!
      try { 
        googleCodingConvention0.applySingletonGetter((FunctionType) null, (FunctionType) null, (ObjectType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ClosureCodingConvention", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      String string0 = closureCodingConvention0.getGlobalObject();
      assertEquals("goog.global", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      Collection<CodingConvention.AssertionFunctionSpec> collection0 = closureCodingConvention0.getAssertionFunctions();
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      CodingConvention.SubclassType codingConvention_SubclassType0 = CodingConvention.SubclassType.MIXIN;
      googleCodingConvention0.applySubclassRelationship((FunctionType) null, (FunctionType) null, codingConvention_SubclassType0);
      assertEquals("goog.exportSymbol", googleCodingConvention0.getExportSymbolFunction());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      CodingConvention.SubclassType codingConvention_SubclassType0 = CodingConvention.SubclassType.INHERITS;
      // Undeclared exception!
      try { 
        googleCodingConvention0.applySubclassRelationship((FunctionType) null, (FunctionType) null, codingConvention_SubclassType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ClosureCodingConvention", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionTypeWithVarArgs((JSType) null, (List<JSType>) arrayList0);
      JSType[] jSTypeArray0 = new JSType[1];
      jSTypeArray0[0] = (JSType) functionType0;
      Node node0 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      CodingConvention.SubclassRelationship codingConvention_SubclassRelationship0 = googleCodingConvention0.getClassesDefinedByCall(node0);
      assertNull(codingConvention_SubclassRelationship0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Node node0 = new Node(33);
      Node node1 = new Node(46, node0, node0, node0, node0);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      // Undeclared exception!
      try { 
        googleCodingConvention0.getClassesDefinedByCall(node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ClosureCodingConvention", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("4", "4");
      CodingConvention.SubclassRelationship codingConvention_SubclassRelationship0 = closureCodingConvention0.getClassesDefinedByCall(node0);
      assertNull(codingConvention_SubclassRelationship0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Node node0 = Node.newNumber(0.0);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      String string0 = googleCodingConvention0.extractClassNameIfProvide(node0, node0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("4", "4");
      List<String> list0 = closureCodingConvention0.identifyTypeDeclarationCall(node0);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Node node0 = new Node(5);
      Node node1 = new Node(50, node0, node0);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      String string0 = googleCodingConvention0.getSingletonGetterClassName(node1);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      Node node0 = new Node((-553), (-553), (-553));
      // Undeclared exception!
      try { 
        closureCodingConvention0.isPropertyTestFunction(node0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Node node0 = new Node(1771);
      Node node1 = new Node(37, node0, node0, node0, node0);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      boolean boolean0 = googleCodingConvention0.isPropertyTestFunction(node1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      Node node0 = new Node(38);
      // Undeclared exception!
      try { 
        closureCodingConvention0.getObjectLiteralCast((NodeTraversal) null, node0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Node node0 = new Node((-1005));
      Node node1 = new Node(37, node0, node0, node0, node0);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      CodingConvention.ObjectLiteralCast codingConvention_ObjectLiteralCast0 = googleCodingConvention0.getObjectLiteralCast((NodeTraversal) null, node1);
      assertNull(codingConvention_ObjectLiteralCast0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Node node0 = new Node(42, 42, 42);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      CodingConvention.Bind codingConvention_Bind0 = googleCodingConvention0.describeFunctionBind(node0);
      assertNull(codingConvention_Bind0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Node node0 = new Node(42, 42, 42);
      Node node1 = new Node(37, node0, node0, 36, 9);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      CodingConvention.Bind codingConvention_Bind0 = googleCodingConvention0.describeFunctionBind(node1);
      assertNull(codingConvention_Bind0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Node node0 = new Node((-1005));
      Node node1 = new Node(37, node0, node0, node0, node0);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      CodingConvention.Bind codingConvention_Bind0 = googleCodingConvention0.describeFunctionBind(node1);
      assertNull(codingConvention_Bind0);
  }
}
/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:20:08 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.GoogleCodingConvention;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("3sGy:8\"6k%:mgmas4.P[", "3sGy:8\"6k%:mgmas4.P[");
      Node node1 = new Node(120, node0, node0, node0, node0, 1418, 1);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("t.R");
      Node node1 = new Node(22, node0, node0, node0, node0, 9, 34);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      Scope scope1 = typedScopeCreator0.createScope(node0, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode("gog.getCssame");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Node node1 = new Node(118, node0, node0, node0, node0, 20, 40);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(VAR):  [testcode] :20:40
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("", "");
      Node node0 = compiler0.parse(jSSourceFile0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Node node1 = new Node(118, node0, node0, node0, node0, 18, 1);
      node1.addSuppression("");
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode("TypeScopeCreator$Disc$verEnums");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Node node1 = new Node(39, node0, node0, node0, node0, 0, 46);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseSyntheticCode("Id generator call must be unconditional", "Id generator call must be unconditional");
      Node node1 = new Node(41, node0, node0, node0, node0, 48, 30);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("3sGy:8\"%:mgmas4.P[", "3sGy:8\"%:mgmas4.P[");
      Node node1 = new Node(43, node0, node0, node0, node0, 38, 31);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("hdR");
      Node node1 = new Node(44, node0, node0, 45, 27);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode(" H'Wq/N");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Node node1 = new Node(47, node0, node0, node0, node0, 707, (-3));
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode("goog.getCssvve");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Node node1 = new Node(64, node0, node0, node0, node0, 16, 23);
      typedScopeCreator0.createScope(node1, (Scope) null);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode("gog.getCssame");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Node node1 = new Node(69, node0, node0, node0, node0, 31, 10);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("Qn", "Qn");
      Node node1 = new Node(122, node0, node0, node0, node0, 34, (-3));
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      Node node0 = jSTypeRegistry0.createParameters((List<JSType>) arrayList0);
      compiler0.toSource(node0);
      Node node1 = new Node(105, node0, node0, node0, node0, 42, 4);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("3sGy:8\"%:mgmas4.P[", "3sGy:8\"%:mgmas4.P[");
      Node node1 = new Node(37, node0, node0, node0, node0, 38, 31);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode("goog.getCssame");
      Node node1 = new Node(86, node0, node0, node0, node0, 38, 1);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("goLg.getCssvv.e");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("", "");
      Node node0 = compiler0.parse(jSSourceFile0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Node node1 = compiler0.parse(jSSourceFile0);
      Node node2 = new Node(118, node0, node0, node0, node1, 18, 1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode("goog.getCssame");
      node0.addSuppression("goog.getCssame");
      Node node1 = new Node(86, node0, node0, node0, node0, 38, 1);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("", "");
      Node node0 = compiler0.parse(jSSourceFile0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[2];
      jSSourceFileArray0[0] = jSSourceFile0;
      jSSourceFileArray0[1] = jSSourceFile0;
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.init(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
      Node node1 = new Node(118, node0, node0, node0, node0, 18, 1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      compiler0.parseTestCode("gog.getCssame");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Node node0 = compiler0.parseSyntheticCode("gog.getCssame", "gog.getCssame");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode("goog.typedef");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseSyntheticCode("3sGy:8\"%:mgmas4.P[", "3sGy:8\"%:mgmas4.P[");
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node1 = jSTypeRegistry0.createParameters((List<JSType>) stack0);
      Node node2 = new Node((-942), node1, node1, node0, node0, 10, 9);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseSyntheticCode("3sGy:8\"%:mgmas4.P[", "3sGy:8\"%:mgmas4.P[");
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node1 = jSTypeRegistry0.createParameters((List<JSType>) stack0);
      Node node2 = new Node((-942), node1, node1, node0, node0, 10, 9);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Node node3 = new Node(4095, node2);
      Scope scope1 = typedScopeCreator0.createScope(node3, scope0);
      assertFalse(scope1.isGlobal());
  }
}
/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:31:10 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
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
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("com.google.javascrpt.jscomp.TypedScopeCreator$1");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(120, 87, 87);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node0, (Scope) null);
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
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(118);
      Node node1 = new Node(47);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope scope1 = typedScopeCreator0.createScope(node1, scope0);
      assertFalse(scope1.isGlobal());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(86, 86, 86);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node0, (Scope) null);
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
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(118, 118, 118);
      SyntheticAst syntheticAst0 = new SyntheticAst("Q7?%G");
      Node node1 = syntheticAst0.getAstRoot(compiler0);
      node0.addChildToBack(node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node0, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(38);
      node0.addSuppression("hasField() can only be called on non-repeated fields.");
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("73");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(41, 41, 41);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(43);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(44, 44, 44);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(47);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(69);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(122, 33, 33);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(64);
      typedScopeCreator0.createScope(node0, (Scope) null);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(105);
      Node node1 = new Node(38);
      Scope scope0 = typedScopeCreator0.createInitialScope(node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node0, scope0);
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
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node((-1207));
      Node node1 = new Node(37, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("goog.typedef");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      Node node1 = new Node((-2630), node0, node0, 2027, 575);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      Node node1 = Node.newString(2, "4']J}", (-1), 29);
      node1.addChildToBack(node0);
      Node node2 = new Node((-2630), node1, node1, 2027, 575);
      Scope scope0 = typedScopeCreator0.createInitialScope(node1);
      Scope scope1 = typedScopeCreator0.createScope(node2, scope0);
      assertFalse(scope1.isGlobal());
  }
}
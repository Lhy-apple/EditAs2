/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:16:41 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.DefaultCodingConvention;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("com.google.commb.base.Priates");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(83);
      Node node1 = new Node(83, node0, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      Scope scope1 = typedScopeCreator0.createScope(node0, scope0);
      assertFalse(scope1.isGlobal());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Node node0 = new Node(105, 105, 105);
      Compiler compiler0 = new Compiler();
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
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
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(118, (-2019), (-2019));
      Node node1 = new Node(18, node0, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
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
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DefaultCodingConvention defaultCodingConvention0 = new DefaultCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, defaultCodingConvention0);
      Node node0 = new Node(37, 37, 37);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(86);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
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
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(118);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
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
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(33);
      Node node1 = new Node(83, node0, node0, node0, node0);
      Node node2 = new Node(1, node1, 27, 14);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(83);
      Node node1 = new Node(83, node0, node0, node0, node0);
      Node node2 = new Node(1, node1, 27, 14);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:49:02 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.GoogleCodingConvention;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.Tracer;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("setZd ", "setZd ");
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parse(jSSourceFile0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: consistentIdGenerator
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFt() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      typedScopeCreator0.patchGlobalScope(scope0, node0);
      assertEquals(33, scope0.getVarCount());
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFt() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = compiler0.parseSyntheticCode("function JSCompiler_identityFt() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}", "function JSCompiler_identityFt() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope scope1 = typedScopeCreator0.createScope(node1, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("qset", "qset");
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parse(jSSourceFile0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: consistentIdGenerator
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ms.setter2parms");
      Node node1 = new Node(86, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("qset", "qset");
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parse(jSSourceFile0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: consistentIdGenerator
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("qset ", "qset ");
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parse(jSSourceFile0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: consistentIdGenerator
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("set ");
      Node node1 = new Node(43, node0, node0, node0, 39, 4095);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ms.setter2parms");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(44, node0, node0, node0, 122, (-5549));
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ms.setter2parms");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(47, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("setZd ", "setZd ");
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parse(jSSourceFile0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: consistentIdGenerator
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ms.setter2parms");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(64, node0);
      Node node2 = new Node(43, node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // Unexpected node type: SCRIPT 1 [synthetic: com.google.javascript.rhino.Node$IntPropListItem@0000000504] [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000505] [input_id: com.google.javascript.rhino.Node$ObjectPropListItem@0000000506]
         //   Node(OBJECTLIT):  [testcode] :-1:-1
         // [source unknown]
         //   Parent(FALSE):  [testcode] :-1:-1
         // [source unknown]
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.googl.javascript.jscomp.TypedScopeCreator$AbstactScopB:ilr$CllectProperties");
      Node node1 = new Node(15);
      Node node2 = new Node(120, node1, node0, node1, 38, 30);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, (Scope) null);
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
      Node node0 = compiler0.parseTestCode("msg.setter2.parms");
      Node node1 = new Node(35, node0, node0, node0, 4095, 2);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node1);
      Scope scope1 = typedScopeCreator0.createScope(node0, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("msg.setter2.parms", "msg.setter2.parms");
      Logger logger0 = Tracer.logger;
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = compiler0.parse(jSSourceFile0);
      compiler0.parseTestCode("msg.setter2.parms");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      typedScopeCreator0.patchGlobalScope(scope0, node0);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.TypedScopeCreator$AbstractScopeBuilder$CollectProperties");
      Node node1 = compiler0.parseTestCode("com.google.javascript.jscomp.TypedScopeCreator$AbstractScopeBuilder$CollectProperties");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node2 = new Node(4095, node1, node0, node0, 0, 3326);
      Scope scope0 = typedScopeCreator0.createScope(node2, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }
}
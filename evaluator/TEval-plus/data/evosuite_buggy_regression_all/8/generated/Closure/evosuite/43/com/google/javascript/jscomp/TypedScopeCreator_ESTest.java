/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:10:33 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.GoogleCodingConvention;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.JqueryCodingConvention;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node0 = Normalize.parseAndNormalizeSyntheticCode(compiler0, "msg.illegal.character", (String) null);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
      
      typedScopeCreator0.patchGlobalScope(scope0, node0);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(37, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(120, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(CATCH): [[singleton]]:-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Node node0 = Normalize.parseAndNormalizeSyntheticCode(compiler0, " [testcode] ", (String) null);
      Node node1 = new Node(64, node0);
      Node node2 = new Node(39, node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, (Scope) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Number node not created with Node.newNumber
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node0 = Normalize.parseAndNormalizeSyntheticCode(compiler0, "com.google.javascript.jscomp.TypedScopeCreator$1", (String) null);
      Node node1 = new Node(47, node0, node0, node0, 4, 2);
      Scope scope0 = typedScopeCreator0.createInitialScope(node1);
      Scope scope1 = typedScopeCreator0.createScope(node0, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(118, node0);
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
         //   Node(VAR): [[singleton]]:-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node0 = Normalize.parseAndNormalizeSyntheticCode(compiler0, "G$=a6;uHDmj%Fs;q*pk", (String) null);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Node node0 = Normalize.parseAndNormalizeSyntheticCode(compiler0, "", (String) null);
      Node node1 = new Node(105, node0);
      Scope scope0 = new Scope(node0, compiler0);
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
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(39, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(41, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(43, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(44, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(47, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Vector<JSSourceFile> vector0 = new Vector<JSSourceFile>();
      compiler0.compile((List<JSSourceFile>) vector0, (List<JSSourceFile>) vector0, compilerOptions0);
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(122, node0);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Vector<JSSourceFile> vector0 = new Vector<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) vector0, (List<JSSourceFile>) vector0, compilerOptions0);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node0 = new Node(2623);
      Node node1 = new Node(118, node0, node0);
      Node node2 = new Node(2623, node1);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
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
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node0 = Normalize.parseAndNormalizeSyntheticCode(compiler0, "ms.illegal.character", (String) null);
      node0.setType(65023);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) linkedList0);
      Node node1 = new Node(30, node0, 50, 4);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) linkedList0);
      Node node1 = new Node(30, node0, 50, 4);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Node node2 = new Node(61, node1, node1);
      Scope scope1 = typedScopeCreator0.createScope(node2, scope0);
      assertFalse(scope1.isGlobal());
  }
}
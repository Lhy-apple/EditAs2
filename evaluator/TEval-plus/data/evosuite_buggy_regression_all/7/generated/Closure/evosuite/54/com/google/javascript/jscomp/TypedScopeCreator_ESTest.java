/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:02:13 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.DefaultPassConfig;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Compiler compiler0 = new Compiler();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.ConcreteType$ConcreteFunctionType");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
      
      typedScopeCreator0.patchGlobalScope(scope0, node0);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Vector<JSSourceFile> vector0 = new Vector<JSSourceFile>();
      compiler0.compile((List<JSSourceFile>) vector0, (List<JSSourceFile>) vector0, compilerOptions0);
      Node node0 = new Node(64, 64, (-1344));
      Node node1 = new Node(37, node0, 42, 2);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      DefaultPassConfig defaultPassConfig0 = new DefaultPassConfig(compilerOptions0);
      Scope scope0 = defaultPassConfig0.getTopScope();
      Node node0 = new Node(44, 44, 44);
      // Undeclared exception!
      try { 
        typedScopeCreator0.patchGlobalScope(scope0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Compiler compiler0 = new Compiler();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      DefaultPassConfig defaultPassConfig0 = compiler0.ensureDefaultPassConfig();
      Scope scope0 = defaultPassConfig0.getTopScope();
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.ConcreteType$ConcreteFunctionType");
      // Undeclared exception!
      try { 
        typedScopeCreator0.patchGlobalScope(scope0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node((-1344), (-1344), (-1344));
      Node node1 = new Node(118, node0, 13, 19);
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
  public void test05()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(105, 92, (-1001));
      Node node1 = new Node(31, node0, 21, 42);
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
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(86, node0, 1057, 0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = Node.newNumber(1826.03370006566, (-1285), 15);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      Node node0 = new Node((-1297), (-1297), 1158);
      Node node1 = new Node(41, node0, 16, 21);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(66, 66, 66);
      Node node1 = new Node(43, node0, node0, 22, 9);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(44, 44, 44);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(47, 1825, 1825);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = Node.newString("");
      Node node1 = new Node(69, node0, node0, 339, 18);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      ArrayList<JSSourceFile> arrayList0 = new ArrayList<JSSourceFile>();
      compiler0.compile((List<JSSourceFile>) arrayList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      DefaultPassConfig defaultPassConfig0 = compiler0.ensureDefaultPassConfig();
      Node node0 = new Node(122, (-2799), (-1344));
      defaultPassConfig0.regenerateGlobalTypedScope(compiler0, node0);
      assertEquals((-1), node0.getSourcePosition());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = Node.newString("'m9q_f2", 64, (-635));
      Node node1 = new Node(64, node0, 40, 36);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(120, node0, 1, (-1297));
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
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseInputs();
      Node node1 = new Node(2271, 28, 14);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      Scope scope1 = typedScopeCreator0.createScope(node0, scope0);
      assertFalse(scope1.isGlobal());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseInputs();
      LinkedList<JSType> linkedList1 = new LinkedList<JSType>();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      Node node1 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) linkedList1);
      Node node2 = new Node(8202, node1, node1, (-1156), 86);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}
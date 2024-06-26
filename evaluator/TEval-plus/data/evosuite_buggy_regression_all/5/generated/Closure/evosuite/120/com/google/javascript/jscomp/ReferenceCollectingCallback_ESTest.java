/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:41:39 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Predicate;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerInput;
import com.google.javascript.jscomp.ConstCheck;
import com.google.javascript.jscomp.MemoizedScopeCreator;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.ReferenceCollectingCallback;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.SyntacticScopeCreator;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.rhino.InputId;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.StaticSourceFile;
import java.util.Iterator;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ReferenceCollectingCallback_ESTest extends ReferenceCollectingCallback_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("com");
      CompilerInput compilerInput0 = new CompilerInput(sourceFile0, true);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.createRefForTest(compilerInput0);
      InputId inputId0 = referenceCollectingCallback_Reference0.getInputId();
      assertEquals("com", inputId0.getIdName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      SourceFile sourceFile0 = SourceFile.fromGenerator("Fv", sourceFile_Generator0);
      CompilerInput compilerInput0 = new CompilerInput(sourceFile0, true);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.createRefForTest(compilerInput0);
      Scope scope0 = referenceCollectingCallback_Reference0.getScope();
      assertNull(scope0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("com");
      CompilerInput compilerInput0 = new CompilerInput(sourceFile0, true);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.createRefForTest(compilerInput0);
      Node node0 = referenceCollectingCallback_Reference0.getGrandparent();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SyntheticAst syntheticAst0 = new SyntheticAst("argument^");
      CompilerInput compilerInput0 = new CompilerInput(syntheticAst0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.createRefForTest(compilerInput0);
      StaticSourceFile staticSourceFile0 = referenceCollectingCallback_Reference0.getSourceFile();
      assertNull(staticSourceFile0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("com");
      CompilerInput compilerInput0 = new CompilerInput(sourceFile0, true);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.createRefForTest(compilerInput0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = referenceCollectingCallback_Reference0.getBasicBlock();
      assertNull(referenceCollectingCallback_BasicBlock0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("arguments");
      Node node1 = new Node(8, node0, node0, node0, 1, 538);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 8
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SyntheticAst syntheticAst0 = new SyntheticAst("arguments");
      CompilerInput compilerInput0 = new CompilerInput(syntheticAst0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.createRefForTest(compilerInput0);
      Scope scope0 = Scope.createGlobalScope((Node) null);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference1 = referenceCollectingCallback_Reference0.cloneWithNewScope(scope0);
      assertNotSame(referenceCollectingCallback_Reference1, referenceCollectingCallback_Reference0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SyntheticAst syntheticAst0 = new SyntheticAst("arguments");
      CompilerInput compilerInput0 = new CompilerInput(syntheticAst0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.createRefForTest(compilerInput0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_Reference0.getSymbol();
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // NAME is not a string node
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SourceFile sourceFile0 = new SourceFile("d");
      CompilerInput compilerInput0 = new CompilerInput(sourceFile0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.createRefForTest(compilerInput0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_Reference0.isVarDeclaration();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      Compiler compiler0 = new Compiler();
      ConstCheck constCheck0 = new ConstCheck((AbstractCompiler) null);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, constCheck0, (ScopeCreator) null);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, ".mMU97&R/3ATA@3");
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      // Undeclared exception!
      try { 
        ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, referenceCollectingCallback_BasicBlock0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SyntheticAst syntheticAst0 = new SyntheticAst("arguments");
      CompilerInput compilerInput0 = new CompilerInput(syntheticAst0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.createRefForTest(compilerInput0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_Reference0.isHoistedFunction();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      Iterator<ReferenceCollectingCallback.Reference> iterator0 = referenceCollectingCallback_ReferenceCollection0.iterator();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("42k2^`^[->4gXz");
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.hotSwapScript(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Predicate<Scope.Var> predicate0 = (Predicate<Scope.Var>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, (ReferenceCollectingCallback.Behavior) null, predicate0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.getScope((Scope.Var) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback((AbstractCompiler) null, referenceCollectingCallback_Behavior0, (Predicate<Scope.Var>) null);
      Iterable<Scope.Var> iterable0 = referenceCollectingCallback0.getAllSymbols();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      Predicate<Scope.Var> predicate0 = (Predicate<Scope.Var>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0, predicate0);
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = referenceCollectingCallback0.getReferences((Scope.Var) null);
      assertNull(referenceCollectingCallback_ReferenceCollection0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Node node0 = Node.newString("m=K", 98, 98);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock1 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      boolean boolean0 = referenceCollectingCallback_BasicBlock1.provablyExecutesBefore(referenceCollectingCallback_BasicBlock0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("msg.let.redecl");
      Node node1 = new Node(109, node0, node0, node0, 39, 39);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 109
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("arguments");
      Node node1 = new Node(4095, node0, node0, node0, 2120, 27);
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      Predicate<Scope.Var> predicate0 = (Predicate<Scope.Var>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      doReturn(false, false).when(predicate0).apply(any(com.google.javascript.jscomp.Scope.Var.class));
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0, predicate0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 4095
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("42k2^`^-->4gXz");
      Node node1 = new Node(115, node0, node0, node0, 53, 53);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(SCRIPT): [testcode]:-1:-1
         // [source unknown]
         //   Parent(FOR): [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("x~| {W3zFV?AK/Z]]");
      Node node1 = new Node(77, node0, node0, node0, (-1080), 77);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(SCRIPT): [testcode]:-1:-1
         // [source unknown]
         //   Parent(TRY): [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("x~| {W3zFV?AK/Z]]");
      Node node1 = new Node(98, node0, node0, node0, (-1080), 98);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(SCRIPT): [testcode]:-1:-1
         // [source unknown]
         //   Parent(HOOK): [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("42k2^`^[->4gXz");
      Node node1 = new Node(101, node0, node0, node0, 39, 39);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(SCRIPT): [testcode]:-1:-1
         // [source unknown]
         //   Parent(AND): [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("42k2^`^-->4gXz");
      Node node1 = new Node(113, node0, node0, node0, 31, (-1010));
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(SCRIPT): [testcode]:-1:-1
         // [source unknown]
         //   Parent(WHILE): [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("instanceof");
      Node node1 = new Node(114, node0, node0, node0, 48, 29);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(SCRIPT): [testcode]:-1:-1
         // [source unknown]
         //   Parent(DO): [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("42k2^`^[->4gXz");
      Node node1 = new Node(119, node0, node0, node0, 53, 53);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(SCRIPT): [testcode]:-1:-1
         // [source unknown]
         //   Parent(WITH): [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseTestCode("0e@");
      Node node1 = new Node(100, node0, node0, node0, 1, 36);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      MemoizedScopeCreator memoizedScopeCreator0 = new MemoizedScopeCreator(syntacticScopeCreator0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0, memoizedScopeCreator0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.visit(nodeTraversal0, node1, node1);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      referenceCollectingCallback_ReferenceCollection0.add((ReferenceCollectingCallback.Reference) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.isWellDefined();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isWellDefined();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isEscaped();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      referenceCollectingCallback_ReferenceCollection0.add((ReferenceCollectingCallback.Reference) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.isEscaped();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = referenceCollectingCallback_ReferenceCollection0.getInitializingReferenceForConstants();
      assertNull(referenceCollectingCallback_Reference0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isAssignedOnceInLifetime();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      referenceCollectingCallback_ReferenceCollection0.add((ReferenceCollectingCallback.Reference) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.isAssignedOnceInLifetime();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isNeverAssigned();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      referenceCollectingCallback_ReferenceCollection0.add((ReferenceCollectingCallback.Reference) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.isNeverAssigned();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.firstReferenceIsAssigningDeclaration();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      referenceCollectingCallback_ReferenceCollection0.add((ReferenceCollectingCallback.Reference) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.firstReferenceIsAssigningDeclaration();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      ConstCheck constCheck0 = new ConstCheck((AbstractCompiler) null);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, ".mMU97&R/3ATA@3");
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock1 = new ReferenceCollectingCallback.BasicBlock(referenceCollectingCallback_BasicBlock0, node0);
      boolean boolean0 = referenceCollectingCallback_BasicBlock1.isGlobalScopeBlock();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Node node0 = Node.newString("m=K", 98, 98);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock1 = new ReferenceCollectingCallback.BasicBlock(referenceCollectingCallback_BasicBlock0, node0);
      boolean boolean0 = referenceCollectingCallback_BasicBlock1.provablyExecutesBefore(referenceCollectingCallback_BasicBlock0);
      assertFalse(boolean0);
  }
}

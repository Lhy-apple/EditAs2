/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:39:35 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.ClosureReverseAbstractInterpreter;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.DefaultCodingConvention;
import com.google.javascript.jscomp.MemoizedScopeCreator;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.RuntimeTypeCheck;
import com.google.javascript.jscomp.SyntacticScopeCreator;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Normalize_ESTest extends Normalize_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.javascript.rhino.SourcePosition", "com.google.javascript.rhino.SourcePosition");
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, false);
      // Undeclared exception!
      try { 
        normalize_VerifyConstants0.process(node0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.parseAndNormalizeTestCode(compiler0, "om.google.javascript.rhino.SourcePosition", "om.google.javascript.rhino.SourcePosition");
      // Undeclared exception!
      try { 
        compiler0.optimize();
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
      // Undeclared exception!
      try { 
        RuntimeTypeCheck.getBoilerplateCode(compiler0, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "om.google.javascript.hino.SurcePsition", "om.google.javascript.hino.SurcePsition");
      Normalize.PropagateConstantAnnotationsOverVars normalize_PropagateConstantAnnotationsOverVars0 = new Normalize.PropagateConstantAnnotationsOverVars(compiler0, false);
      // Undeclared exception!
      try { 
        normalize_PropagateConstantAnnotationsOverVars0.process(node0, node0);
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
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "om.google.javascript.hino.SurcePsition", "om.google.javascript.hino.SurcePsition");
      Normalize.PropagateConstantAnnotationsOverVars normalize_PropagateConstantAnnotationsOverVars0 = new Normalize.PropagateConstantAnnotationsOverVars(compiler0, false);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_PropagateConstantAnnotationsOverVars0);
      assertFalse(node0.isQuotedString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "FMY", "FMY");
      Node node1 = compiler0.parseTestCode("FMY");
      node0.addChildToBack(node1);
      Normalize.PropagateConstantAnnotationsOverVars normalize_PropagateConstantAnnotationsOverVars0 = new Normalize.PropagateConstantAnnotationsOverVars(compiler0, true);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_PropagateConstantAnnotationsOverVars0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // Unexpected const change.
         //   name: FMY
         //   parent:EXPR_RESULT 1 [sourcename: java.lang.String@0000000435]
         //     NAME FMY 1 [sourcename: java.lang.String@0000000435]
         // 
         //   Node(NAME FMY):  [testcode] :1:0
         // [source unknown]
         //   Parent(EXPR_RESULT):  [testcode] :1:0
         // [source unknown]
         //
         verifyException("com.google.javascript.jscomp.Normalize$PropagateConstantAnnotationsOverVars", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.javascript.rhino.SourcePosition", "com.google.javascript.rhino.SourcePosition");
      Node node1 = new Node(24, node0, node0, node0);
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      normalize_VerifyConstants0.process(node0, node0);
      assertEquals(8, Node.FLAG_NO_THROWS);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.parseAndNormalizeTestCode(compiler0, "om.google.javascript.rhino.SourcePosition", "om.google.javascript.rhino.SourcePosition");
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, false);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      JSType[] jSTypeArray0 = new JSType[7];
      DefaultCodingConvention defaultCodingConvention0 = new DefaultCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(defaultCodingConvention0, jSTypeRegistry0);
      JSType jSType0 = closureReverseAbstractInterpreter0.getRestrictedByTypeOfResult((JSType) null, "a", true);
      jSTypeArray0[6] = jSType0;
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs(jSTypeArray0);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_VerifyConstants0);
      assertEquals(1, Node.LEFT);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "om.google.javascript.hiJo.SurcePsition", "om.google.javascript.hiJo.SurcePsition");
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, false);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_VerifyConstants0);
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "FY", "FY");
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_VerifyConstants0);
      assertEquals(46, Node.IS_DISPATCHER);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "w$:fVS-jjd", "w$:fVS-jjd");
      assertFalse(node0.hasMoreThanOneChild());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "w-\"jF>Ix\"", "?5\"Sj,\"OeW");
      assertEquals(0, node0.getCharno());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "FY", "FY");
      Normalize.NormalizeStatements normalize_NormalizeStatements0 = new Normalize.NormalizeStatements((AbstractCompiler) null, true);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_NormalizeStatements0);
      assertEquals(1, Node.TARGET_PROP);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.javascript.rhino.SourcePosition", "com.google.javascript.rhino.SourcePosition");
      Node node1 = new Node(24, node0, node0, node0);
      Node node2 = Node.newString(105, "com.google.javascript.rhino.SourcePosition", 4095, 37);
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      node1.addChildToBack(node2);
      MemoizedScopeCreator memoizedScopeCreator0 = new MemoizedScopeCreator(syntacticScopeCreator0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, normalize_VerifyConstants0, memoizedScopeCreator0);
      Normalize.NormalizeStatements normalize_NormalizeStatements0 = new Normalize.NormalizeStatements(compiler0, true);
      normalize_NormalizeStatements0.visit(nodeTraversal0, node2, node2);
      assertEquals(0, node2.getChildCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.javascript.rhino.SourcePosition", "com.google.javascript.rhino.SourcePosition");
      Node node1 = Node.newString(105, "com.google.javascript.rhino.SourcePosition", 4095, 37);
      node0.addChildToBack(node1);
      Normalize.NormalizeStatements normalize_NormalizeStatements0 = new Normalize.NormalizeStatements(compiler0, true);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_NormalizeStatements0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}
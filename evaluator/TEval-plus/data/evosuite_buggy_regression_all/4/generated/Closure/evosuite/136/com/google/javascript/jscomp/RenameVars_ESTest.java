/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:23:04 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerInput;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.GatherRawExports;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.LightweightMessageFormatter;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.NameAnonymousFunctionsMapped;
import com.google.javascript.jscomp.NameReferenceGraph;
import com.google.javascript.jscomp.RenameVars;
import com.google.javascript.jscomp.SourceExcerptProvider;
import com.google.javascript.jscomp.VariableMap;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.ScriptOrFnNode;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.nio.charset.Charset;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RenameVars_ESTest extends RenameVars_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      char[] charArray0 = new char[1];
      RenameVars renameVars0 = new RenameVars(compiler0, "", false, false, false, (VariableMap) null, charArray0, compilerOptions0.stripNameSuffixes);
      VariableMap variableMap0 = renameVars0.getVariableMap();
      assertNotNull(variableMap0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[1];
      GatherRawExports gatherRawExports0 = new GatherRawExports(compiler0);
      Set<String> set0 = gatherRawExports0.getExportedVariableNames();
      RenameVars renameVars0 = new RenameVars(compiler0, "com.google.javascript.jscomp.RenameVars$2", false, false, false, (VariableMap) null, charArray0, set0);
      RenameVars.Assignment renameVars_Assignment0 = renameVars0.new Assignment("o6U$AEh1dt", (CompilerInput) null);
      renameVars_Assignment0.newName = ">/V75?u-ES=P\"qSorZi";
      // Undeclared exception!
      try { 
        renameVars_Assignment0.setNewName(" => ");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[0];
      RenameVars renameVars0 = new RenameVars(compiler0, (String) null, true, true, true, (VariableMap) null, charArray0, (Set<String>) null);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      HashSet<String> hashSet0 = new HashSet<String>();
      char[] charArray0 = new char[6];
      RenameVars renameVars0 = new RenameVars(compiler0, "co", false, false, false, (VariableMap) null, charArray0, hashSet0);
      Node node0 = compiler0.parseSyntheticCode(" => ", "Xk");
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[1];
      NameReferenceGraph nameReferenceGraph0 = new NameReferenceGraph(compiler0);
      NameReferenceGraph.Name nameReferenceGraph_Name0 = nameReferenceGraph0.new Name("L v6\"Li}}9SdRyfG", false);
      JSType jSType0 = nameReferenceGraph_Name0.getType();
      jSTypeArray0[0] = jSType0;
      Node node1 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      renameVars0.process(node0, node1);
      assertFalse(node0.wasEmptyNode());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      char[] charArray0 = new char[6];
      RenameVars renameVars0 = new RenameVars(compiler0, " => ", false, true, true, (VariableMap) null, charArray0, linkedHashSet0);
      Node node0 = compiler0.parseSyntheticCode("com.google.common.collect.Multmaps$UnmodifiableSetMultimap", "com.google.common.collect.Multmaps$UnmodifiableSetMultimap");
      // Undeclared exception!
      try { 
        renameVars0.process(node0, node0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // prefix must start with one of: [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, $]
         //
         verifyException("com.google.javascript.jscomp.NameGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      SourceExcerptProvider.SourceExcerpt sourceExcerptProvider_SourceExcerpt0 = SourceExcerptProvider.SourceExcerpt.LINE;
      LightweightMessageFormatter lightweightMessageFormatter0 = new LightweightMessageFormatter((SourceExcerptProvider) null, sourceExcerptProvider_SourceExcerpt0);
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      char[] charArray0 = new char[3];
      GatherRawExports gatherRawExports0 = new GatherRawExports(compiler0);
      Set<String> set0 = gatherRawExports0.getExportedVariableNames();
      RenameVars renameVars0 = new RenameVars(compiler0, "g=> ", true, false, true, (VariableMap) null, charArray0, set0);
      RenameVars.ProcessVars renameVars_ProcessVars0 = renameVars0.new ProcessVars(false);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("com.google.javascript.jscomp.Normalize$PropogateConstantAnnotations", "vp{MVH`ppEe;%wN9c");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0);
      renameVars_ProcessVars0.incCount("g=> ", compilerInput0);
      renameVars_ProcessVars0.incCount("g=> ", compilerInput0);
      assertEquals("vp{MVH`ppEe;%wN9c", compilerInput0.getCode());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnonymousFunctionsMapped nameAnonymousFunctionsMapped0 = new NameAnonymousFunctionsMapped(compiler0);
      VariableMap variableMap0 = nameAnonymousFunctionsMapped0.getFunctionMap();
      HashSet<String> hashSet0 = new HashSet<String>();
      char[] charArray0 = new char[1];
      RenameVars renameVars0 = new RenameVars(compiler0, "", false, false, false, variableMap0, charArray0, hashSet0);
      Node node0 = new Node(315);
      RenameVars.ProcessVars renameVars_ProcessVars0 = renameVars0.new ProcessVars(false);
      Charset charset0 = Charset.defaultCharset();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("", charset0);
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0);
      renameVars_ProcessVars0.incCount("J^?TV.:~~", compilerInput0);
      Node node1 = compiler0.parseSyntheticCode("", "_dcp");
      renameVars0.process(node1, node1);
      renameVars0.process(node1, node0);
      assertEquals(1, Node.DECR_FLAG);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnonymousFunctionsMapped nameAnonymousFunctionsMapped0 = new NameAnonymousFunctionsMapped(compiler0);
      VariableMap variableMap0 = nameAnonymousFunctionsMapped0.getFunctionMap();
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      char[] charArray0 = new char[0];
      RenameVars renameVars0 = new RenameVars(compiler0, "", true, false, false, variableMap0, charArray0, compilerOptions0.stripTypes);
      RenameVars.ProcessVars renameVars_ProcessVars0 = renameVars0.new ProcessVars(true);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("// Input %num%", (Charset) null);
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0);
      renameVars_ProcessVars0.incCount("L com.google.protobuf.GeneratedMessage$FieldAccessorTable$RepeatedFieldAccessor", compilerInput0);
      Node node0 = compiler0.parseSyntheticCode("}", "sk1+%WD3P?D|c74)");
      renameVars0.process(node0, node0);
      assertEquals((-3), Node.LOCAL_BLOCK_PROP);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[1];
      Locale locale0 = Locale.ROOT;
      Set<String> set0 = locale0.getUnicodeLocaleAttributes();
      RenameVars renameVars0 = new RenameVars(compiler0, "IoolvtxAI", true, false, true, (VariableMap) null, charArray0, set0);
      RenameVars.ProcessVars renameVars_ProcessVars0 = renameVars0.new ProcessVars(false);
      Charset charset0 = Charset.defaultCharset();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("IoolvtxAI", charset0);
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0);
      renameVars_ProcessVars0.incCount("$$", compilerInput0);
      renameVars_ProcessVars0.incCount("com.google.protobuf.GeneratedMessage$FieldAccessorTable$RepeatedFieldAccessor", compilerInput0);
      ScriptOrFnNode scriptOrFnNode0 = (ScriptOrFnNode)compiler0.parseSyntheticCode("$$", "com.google.protobuf.GeneratedMessage$FieldAccessorTable$RepeatedFieldAccessor");
      renameVars0.process(scriptOrFnNode0, scriptOrFnNode0);
      assertEquals(0, scriptOrFnNode0.getRegexpCount());
  }
}

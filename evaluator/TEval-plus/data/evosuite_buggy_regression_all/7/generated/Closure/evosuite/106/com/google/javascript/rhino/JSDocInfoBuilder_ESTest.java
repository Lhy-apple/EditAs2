/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:14:34 GMT 2023
 */

package com.google.javascript.rhino;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.JSDocInfoBuilder;
import com.google.javascript.rhino.JSTypeExpression;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.EnumType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.RecordType;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JSDocInfoBuilder_ESTest extends JSDocInfoBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.isConstructorRecorded();
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      jSDocInfoBuilder0.hasParameter("lastuse");
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.isInterfaceRecorded();
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      boolean boolean0 = jSDocInfoBuilder0.isPopulatedWithFileOverview();
      assertFalse(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordConstancy();
      boolean boolean0 = jSDocInfoBuilder0.isPopulatedWithFileOverview();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.recordFileOverview("/v");
      boolean boolean0 = jSDocInfoBuilder0.isPopulatedWithFileOverview();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      boolean boolean0 = jSDocInfoBuilder0.isDescriptionRecorded();
      assertFalse(boolean0);
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      jSDocInfoBuilder0.recordDescription("");
      boolean boolean0 = jSDocInfoBuilder0.isDescriptionRecorded();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      JSDocInfo jSDocInfo0 = jSDocInfoBuilder0.build((String) null);
      assertNull(jSDocInfo0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      JSDocInfo.Visibility jSDocInfo_Visibility0 = JSDocInfo.Visibility.PROTECTED;
      jSDocInfoBuilder0.recordVisibility(jSDocInfo_Visibility0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      
      JSDocInfo jSDocInfo0 = jSDocInfoBuilder0.build((String) null);
      assertEquals(JSDocInfo.Visibility.PROTECTED, jSDocInfo0.getVisibility());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.recordFileOverview("");
      assertTrue(jSDocInfoBuilder0.isPopulatedWithFileOverview());
      
      jSDocInfoBuilder0.build("");
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      jSDocInfoBuilder0.markAnnotation("p", 226, 226);
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.markAnnotation(";~(37j=0fU].(Rb", 0, 0);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      Node node0 = jSTypeRegistry0.createParameters((List<JSType>) linkedList0);
      jSDocInfoBuilder0.markTypeNode(node0, 9, 1014, (-1202), true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.markText("v(R-yh\"`4,>k54%`", 3215, 3215, 3215, 3215);
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.markAnnotation("", (-138), 1);
      jSDocInfoBuilder0.markText("", 0, 0, (-138), 48);
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      Node node0 = new Node(1);
      jSDocInfoBuilder0.markTypeNode(node0, 1, 43, 26, false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.markName("p", 2483, 2483);
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.markAnnotation("com.google.javascript.rhino.JSDocInfoBuilder", (-1095), (-1095));
      jSDocInfoBuilder0.markName(">~:IG", (-1095), (-1095));
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      jSDocInfoBuilder0.recordBlockDescription("DIV");
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordBlockDescription("^WT@7(JaU26rQ");
      assertTrue(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      JSDocInfo.Visibility jSDocInfo_Visibility0 = JSDocInfo.Visibility.INHERITED;
      jSDocInfoBuilder0.recordVisibility(jSDocInfo_Visibility0);
      boolean boolean0 = jSDocInfoBuilder0.recordVisibility(jSDocInfo_Visibility0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, ":4Kc(>?", (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordParameter(":4Kc(>?", jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordParameter("", (JSTypeExpression) null);
      boolean boolean0 = jSDocInfoBuilder0.recordParameter("", (JSTypeExpression) null);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordParameterDescription((String) null, "");
      boolean boolean0 = jSDocInfoBuilder0.recordParameterDescription((String) null, "3$");
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordTemplateTypeName("");
      boolean boolean0 = jSDocInfoBuilder0.recordTemplateTypeName("");
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, (String) null, (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordThrowType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      boolean boolean0 = jSDocInfoBuilder0.recordThrowType((JSTypeExpression) null);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordThrowDescription((JSTypeExpression) null, "com.google.javascript.rhino.JSDocInfoBuilder");
      boolean boolean0 = jSDocInfoBuilder0.recordThrowDescription((JSTypeExpression) null, (String) null);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      boolean boolean0 = jSDocInfoBuilder0.addAuthor(";~(37j=0fU].(Rb");
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      boolean boolean0 = jSDocInfoBuilder0.addReference(";~(37j=0fU].(Rb");
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordVersion("m");
      boolean boolean0 = jSDocInfoBuilder0.recordVersion("m");
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordDeprecationReason("");
      boolean boolean0 = jSDocInfoBuilder0.recordDeprecationReason("");
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, JSType> hashMap0 = new HashMap<String, JSType>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      EnumType enumType0 = jSTypeRegistry0.createEnumType("", recordType0);
      Set<String> set0 = enumType0.getElements();
      jSDocInfoBuilder0.recordSuppressions(set0);
      boolean boolean0 = jSDocInfoBuilder0.recordSuppressions(set0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      boolean boolean0 = jSDocInfoBuilder0.recordType((JSTypeExpression) null);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "_?c", (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordParameter("_?c", jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      boolean boolean0 = jSDocInfoBuilder0.recordTypedef((JSTypeExpression) null);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "com.google.common.collect.Iterators$2", (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordBaseType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordTypedef(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      boolean boolean0 = jSDocInfoBuilder0.recordReturnType((JSTypeExpression) null);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "com.google.javascript.rhino.jstype.ErrorFunctionType", (JSTypeRegistry) null);
      jSDocInfoBuilder0.recordReturnType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordReturnType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, ":4Kc(>?", (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordReturnType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordReturnDescription("F4UT}UA1dn");
      boolean boolean0 = jSDocInfoBuilder0.recordReturnDescription("3+0**8c<[|`j*8S");
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      boolean boolean0 = jSDocInfoBuilder0.recordDefineType((JSTypeExpression) null);
      assertFalse(boolean0);
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "M(dV", (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      boolean boolean0 = jSDocInfoBuilder0.recordEnumParameterType((JSTypeExpression) null);
      assertFalse(boolean0);
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "gS4Y241iXk", (JSTypeRegistry) null);
      jSDocInfoBuilder0.recordThisType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordEnumParameterType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      boolean boolean0 = jSDocInfoBuilder0.recordThisType((JSTypeExpression) null);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "[&@+qq[=", (JSTypeRegistry) null);
      jSDocInfoBuilder0.recordEnumParameterType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordThisType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "C$f,6,MA7)aSx", (JSTypeRegistry) null);
      jSDocInfoBuilder0.recordThisType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordThisType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      boolean boolean0 = jSDocInfoBuilder0.recordBaseType((JSTypeExpression) null);
      assertFalse(boolean0);
      assertFalse(jSDocInfoBuilder0.isPopulated());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "[&@+qq[=", (JSTypeRegistry) null);
      jSDocInfoBuilder0.recordEnumParameterType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordBaseType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      Node node0 = Node.newString(1, "com.google.javascript.rhino.JSDocInfoBuilder");
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression(node0, "", jSTypeRegistry0);
      jSDocInfoBuilder0.recordBaseType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordBaseType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordConstancy();
      boolean boolean0 = jSDocInfoBuilder0.recordConstancy();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      boolean boolean0 = jSDocInfoBuilder0.recordDescription((String) null);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.recordDescription("n");
      boolean boolean0 = jSDocInfoBuilder0.recordDescription("com.google.javascript.rhino.Node");
      assertTrue(jSDocInfoBuilder0.isDescriptionRecorded());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.recordFileOverview("");
      boolean boolean0 = jSDocInfoBuilder0.recordFileOverview("");
      assertTrue(jSDocInfoBuilder0.isPopulatedWithFileOverview());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordHiddenness();
      boolean boolean0 = jSDocInfoBuilder0.recordHiddenness();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordNoTypeCheck();
      boolean boolean0 = jSDocInfoBuilder0.recordNoTypeCheck();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "", jSTypeRegistry0);
      jSDocInfoBuilder0.recordType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordConstructor();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.recordConstructor();
      boolean boolean0 = jSDocInfoBuilder0.recordConstructor();
      assertTrue(jSDocInfoBuilder0.isConstructorRecorded());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.recordInterface();
      boolean boolean0 = jSDocInfoBuilder0.recordConstructor();
      assertTrue(jSDocInfoBuilder0.isInterfaceRecorded());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordPreserveTry();
      boolean boolean0 = jSDocInfoBuilder0.recordPreserveTry();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordOverride();
      boolean boolean0 = jSDocInfoBuilder0.recordOverride();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordNoAlias();
      boolean boolean0 = jSDocInfoBuilder0.recordNoAlias();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordDeprecated();
      boolean boolean0 = jSDocInfoBuilder0.recordDeprecated();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "", (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordInterface();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      jSDocInfoBuilder0.recordConstructor();
      boolean boolean0 = jSDocInfoBuilder0.recordInterface();
      assertTrue(jSDocInfoBuilder0.isConstructorRecorded());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      jSDocInfoBuilder0.recordInterface();
      boolean boolean0 = jSDocInfoBuilder0.recordInterface();
      assertTrue(jSDocInfoBuilder0.isInterfaceRecorded());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordExport();
      boolean boolean0 = jSDocInfoBuilder0.recordExport();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordNoShadow();
      boolean boolean0 = jSDocInfoBuilder0.recordNoShadow();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordImplicitCast();
      boolean boolean0 = jSDocInfoBuilder0.recordImplicitCast();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordNoSideEffects();
      boolean boolean0 = jSDocInfoBuilder0.recordNoSideEffects();
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordImplementedInterface((JSTypeExpression) null);
      boolean boolean0 = jSDocInfoBuilder0.recordImplementedInterface((JSTypeExpression) null);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "62gJWL#", (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      jSDocInfoBuilder0.recordConstructor();
      boolean boolean0 = jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isConstructorRecorded());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "", (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      jSDocInfoBuilder0.recordInterface();
      boolean boolean0 = jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isInterfaceRecorded());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "", (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(true);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordReturnType(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      JSTypeExpression jSTypeExpression0 = new JSTypeExpression((Node) null, "K+1\"rD/V%", (JSTypeRegistry) null);
      JSDocInfoBuilder jSDocInfoBuilder0 = new JSDocInfoBuilder(false);
      assertFalse(jSDocInfoBuilder0.isPopulated());
      
      jSDocInfoBuilder0.recordTypedef(jSTypeExpression0);
      boolean boolean0 = jSDocInfoBuilder0.recordDefineType(jSTypeExpression0);
      assertTrue(jSDocInfoBuilder0.isPopulated());
      assertFalse(boolean0);
  }
}
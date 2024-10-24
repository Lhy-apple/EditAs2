/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:08:18 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.EnumElementType;
import com.google.javascript.rhino.jstype.EnumType;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.IndexedType;
import com.google.javascript.rhino.jstype.InstanceObjectType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NamedType;
import com.google.javascript.rhino.jstype.NoResolvedType;
import com.google.javascript.rhino.jstype.NoType;
import com.google.javascript.rhino.jstype.NumberType;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.RecordType;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import com.google.javascript.rhino.jstype.TemplateType;
import com.google.javascript.rhino.jstype.Visitor;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectType_ESTest extends ObjectType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, true);
      NoType noType0 = new NoType(jSTypeRegistry0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, noType0, false);
      Node node0 = Node.newString(1, "Named type with empty name component");
      boolean boolean0 = instanceObjectType0.defineProperty("Named type with empty name component", noType0, false, (Node) null);
      assertTrue(boolean0);
      
      noType0.setSource(node0);
      assertFalse(noType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      Node node0 = Node.newString(1, "GEnr`1rY#+1i?TZ/v+");
      ObjectType.Property objectType_Property0 = new ObjectType.Property("Not declared as a type name", noResolvedType0, false, node0);
      ObjectType.Property objectType_Property1 = objectType_Property0.getSymbol();
      assertFalse(objectType_Property1.isTypeInferred());
      assertFalse(noResolvedType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectType.Property objectType_Property0 = new ObjectType.Property("=n", (JSType) null, false, (Node) null);
      objectType_Property0.getJSDocInfo();
      assertFalse(objectType_Property0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      Node node0 = Node.newString(1, "Named type with empty name component", 0, 1);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(recordType0, node0);
      hashMap0.put("", recordTypeBuilder_RecordProperty0);
      RecordType recordType1 = jSTypeRegistry0.createRecordType(hashMap0);
      ObjectType.Property objectType_Property0 = recordType1.getOwnSlot("");
      assertNotNull(objectType_Property0);
      
      objectType_Property0.setJSDocInfo((JSDocInfo) null);
      assertFalse(recordType1.isFunctionPrototypeType());
      assertFalse(objectType_Property0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      JSType jSType0 = errorFunctionType0.resolve(simpleErrorReporter0, errorFunctionType0);
      assertFalse(jSType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, true);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      recordType0.getPossibleToBooleanOutcomes();
      assertFalse(recordType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Named type with empty name component", "Unknown class name", 160, (-1));
      namedType0.getIndexType();
      assertFalse(namedType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      boolean boolean0 = recordType0.isObject();
      assertFalse(recordType0.isFunctionPrototypeType());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      recordType0.getPropertyNames();
      assertFalse(recordType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "cL[Qo>/U%DK.I<<jQuD");
      templateType0.getOwnerFunction();
      assertFalse(templateType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "cL[Qo>/U%DK.I<<jQuD");
      boolean boolean0 = templateType0.hasReferenceName();
      assertFalse(templateType0.isFunctionPrototypeType());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      recordType0.getRootNode();
      assertFalse(recordType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "Not declared as a type name");
      boolean boolean0 = templateType0.removeProperty("Not declared as a constructor");
      assertFalse(boolean0);
      assertFalse(templateType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, true);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", 316, 316);
      namedType0.getOwnPropertyNames();
      assertFalse(namedType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "cL[Qo>/U%DK.I<<jQuD");
      templateType0.getParameterType();
      assertFalse(templateType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      noResolvedType0.setJSDocInfo((JSDocInfo) null);
      assertFalse(noResolvedType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      noResolvedType0.clearCachedValues();
      assertFalse(noResolvedType0.isFunctionPrototypeType());
      assertFalse(noResolvedType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Not declared as a type name", "Unknown class name", 1, 0);
      namedType0.getCtorImplementedInterfaces();
      assertFalse(namedType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "5v^vX7oI^k");
      ObjectType objectType0 = FunctionType.getTopDefiningInterface(templateType0, "com.loogle.javascript]rhiny.jstype.SimplaSlot");
      assertFalse(objectType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Not declared as a constructor", "Not declared as a constructor", 1, 0);
      namedType0.getOwnPropertyJSDocInfo((String) null);
      assertFalse(namedType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      IndexedType indexedType0 = new IndexedType(jSTypeRegistry0, recordType0, recordType0);
      Visitor<EnumType> visitor0 = (Visitor<EnumType>) mock(Visitor.class, new ViolatedAssumptionAnswer());
      doReturn((EnumType) null).when(visitor0).caseObjectType(any(com.google.javascript.rhino.jstype.ObjectType.class));
      indexedType0.visit(visitor0);
      assertFalse(indexedType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "cL[Qo>/U%DK.I<<jQuD");
      boolean boolean0 = templateType0.isPropertyInExterns("Y");
      assertFalse(templateType0.isFunctionPrototypeType());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoType noType0 = new NoType(jSTypeRegistry0);
      Node node0 = Node.newString(0, "(`7", (-2767), 0);
      EnumType enumType0 = jSTypeRegistry0.createEnumType("Unknown class name", node0, noType0);
      String string0 = enumType0.getDisplayName();
      assertFalse(enumType0.isFunctionPrototypeType());
      assertEquals("Unknown class name", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, true);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      recordType0.getTypeOfThis();
      assertFalse(recordType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      String string0 = ObjectType.createDelegateSuffix("Trz");
      assertEquals("(Trz)", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, noResolvedType0, true);
      instanceObjectType0.getPropertyNode("Unknown class name");
      assertFalse(instanceObjectType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, true);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      jSTypeRegistry0.resolveTypesInScope(recordType0);
      assertFalse(recordType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      JSTypeNative jSTypeNative0 = JSTypeNative.NO_RESOLVED_TYPE;
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      objectType0.getOwnSlot("Unknown class name");
      assertFalse(objectType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      recordType0.getJSDocInfo();
      assertFalse(recordType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      boolean boolean0 = recordType0.detectImplicitPrototypeCycle();
      assertFalse(boolean0);
      assertFalse(recordType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      String string0 = recordType0.getNormalizedReferenceName();
      assertFalse(recordType0.isFunctionPrototypeType());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "IsInstanceOf(");
      String string0 = templateType0.getNormalizedReferenceName();
      assertEquals("IsInstanceOf", string0);
      assertFalse(templateType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NumberType numberType0 = new NumberType(jSTypeRegistry0);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionTypeWithVarArgs((JSType) numberType0, (List<JSType>) linkedList0);
      EnumType enumType0 = jSTypeRegistry0.createEnumType("", (Node) null, functionType0);
      JSType.TypePair jSType_TypePair0 = enumType0.getTypesUnderEquality(numberType0);
      assertNotNull(jSType_TypePair0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      EnumElementType enumElementType0 = new EnumElementType(jSTypeRegistry0, noResolvedType0, "Unknown class name");
      enumElementType0.testForEquality(noResolvedType0);
      assertTrue(noResolvedType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, true);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      Node node0 = jSTypeRegistry0.createParameters((List<JSType>) linkedList0);
      EnumType enumType0 = jSTypeRegistry0.createEnumType("Unknown class name", node0, recordType0);
      enumType0.getTypesUnderEquality(recordType0);
      assertTrue(recordType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      recordType0.defineInferredProperty("BZ^pN4gAJE/", noResolvedType0, (Node) null);
      assertTrue(recordType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      recordType0.findPropertyType(":L\u0002[W9STkUtNEH`");
      assertTrue(recordType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoType noType0 = new NoType(jSTypeRegistry0);
      JSType jSType0 = noType0.findPropertyType("Trz");
      assertFalse(jSType0.isBooleanValueType());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "MF%%+>.hTlhGv/e");
      FunctionType functionType0 = errorFunctionType0.getSuperClassConstructor();
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, functionType0);
      boolean boolean0 = errorFunctionType0.isImplicitPrototype(instanceObjectType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, noResolvedType0, false);
      boolean boolean0 = instanceObjectType0.hasProperty("Not declared as a constructor");
      assertTrue(noResolvedType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "Unknown class name");
      IndexedType indexedType0 = new IndexedType(jSTypeRegistry0, noResolvedType0, templateType0);
      Node node0 = Node.newString(1, "GEnr`1rY#+1i?TZ/v+");
      ObjectType.Property objectType_Property0 = new ObjectType.Property("Not declared as a type name", indexedType0, false, node0);
      ObjectType.Property objectType_Property1 = objectType_Property0.getDeclaration();
      assertFalse(objectType_Property1.isTypeInferred());
      assertNotNull(objectType_Property1);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      ObjectType.Property objectType_Property0 = new ObjectType.Property("BZ^pN4gAJE/", noResolvedType0, false, (Node) null);
      ObjectType.Property objectType_Property1 = objectType_Property0.getDeclaration();
      assertNull(objectType_Property1);
      assertFalse(objectType_Property0.isTypeInferred());
  }
}

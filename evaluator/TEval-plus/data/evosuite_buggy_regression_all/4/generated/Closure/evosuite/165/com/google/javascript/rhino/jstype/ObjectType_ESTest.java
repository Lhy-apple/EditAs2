/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:26:09 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.AllType;
import com.google.javascript.rhino.jstype.BooleanLiteralSet;
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
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.ParameterizedType;
import com.google.javascript.rhino.jstype.PrototypeObjectType;
import com.google.javascript.rhino.jstype.ProxyObjectType;
import com.google.javascript.rhino.jstype.TemplateType;
import com.google.javascript.rhino.jstype.UnionType;
import com.google.javascript.rhino.jstype.UnknownType;
import com.google.javascript.rhino.jstype.Visitor;
import com.google.javascript.rhino.jstype.VoidType;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectType_ESTest extends ObjectType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      VoidType voidType0 = new VoidType(jSTypeRegistry0);
      EnumElementType enumElementType0 = new EnumElementType(jSTypeRegistry0, voidType0, "Not declared as a type name");
      JSType jSType0 = jSTypeRegistry0.createOptionalType(enumElementType0);
      Node node0 = new Node((-337), 0, (-337));
      EnumType enumType0 = new EnumType(jSTypeRegistry0, "", node0, jSType0);
      enumType0.defineElement("Unknown class name", node0);
      boolean boolean0 = enumType0.defineProperty("Unknown class name", voidType0, true, node0);
      assertTrue(enumElementType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ObjectType objectType0 = jSTypeRegistry0.createNativeAnonymousObjectType();
      JSType[] jSTypeArray0 = new JSType[8];
      jSTypeArray0[7] = (JSType) objectType0;
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs(jSTypeArray0);
      ObjectType.Property objectType_Property0 = new ObjectType.Property("Not declared as a type name", objectType0, true, node0);
      objectType_Property0.getJSDocInfo();
      assertTrue(objectType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      JSType jSType0 = errorFunctionType0.resolveInternal(simpleErrorReporter0, noResolvedType0);
      assertFalse(jSType0.isRecordType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ObjectType objectType0 = jSTypeRegistry0.createNativeAnonymousObjectType();
      JSType jSType0 = objectType0.getRestrictedTypeGivenToBooleanOutcome(true);
      assertEquals(BooleanLiteralSet.TRUE, jSType0.getPossibleToBooleanOutcomes());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Unknown class name", "@[']D/Irc~gx<\"jjpU*", 1, 0);
      JSType jSType0 = namedType0.getIndexType();
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      JSTypeNative jSTypeNative0 = JSTypeNative.REFERENCE_ERROR_TYPE;
      JSType jSType0 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      JSType[] jSTypeArray0 = new JSType[0];
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType(jSType0, false, jSTypeArray0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, functionType0);
      boolean boolean0 = instanceObjectType0.defineInferredProperty("'", functionType0, (Node) null);
      assertTrue(boolean0);
      assertFalse(instanceObjectType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, (String) null);
      templateType0.setPropertyJSDocInfo("Not declared as a constructor", (JSDocInfo) null);
      assertFalse(templateType0.isFunctionType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      boolean boolean0 = noResolvedType0.isObject();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Unknown class name", "@[']D/Irc~gx<\"jjpU*", 1, 0);
      Set<String> set0 = namedType0.getPropertyNames();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "b`eWA{`EdqOfZcaNWL", "can't use .skipNulls() with maps", 0, 0);
      FunctionType functionType0 = namedType0.getOwnerFunction();
      assertNull(functionType0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "H1bojjW`H+TF,lH");
      boolean boolean0 = templateType0.hasReferenceName();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createNativeAnonymousObjectType();
      Node node0 = objectType0.getRootNode();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "PF o[zv Aq");
      EnumElementType enumElementType0 = new EnumElementType(jSTypeRegistry0, errorFunctionType0, "Named type with empty name component");
      boolean boolean0 = enumElementType0.removeProperty("Not declared as a type name");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "Named type with empty name component");
      Set<String> set0 = templateType0.getOwnPropertyNames();
      assertFalse(set0.contains("Named type with empty name component"));
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NamedType namedType0 = (NamedType)jSTypeRegistry0.createNamedType(")", ")", 0, (-599));
      JSType jSType0 = namedType0.getParameterType();
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      noResolvedType0.setJSDocInfo((JSDocInfo) null);
      assertTrue(noResolvedType0.hasInstanceType());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      noResolvedType0.clearCachedValues();
      assertFalse(noResolvedType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      UnknownType unknownType0 = new UnknownType(jSTypeRegistry0, false);
      Iterable<ObjectType> iterable0 = unknownType0.getCtorImplementedInterfaces();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "Named type with empty name component");
      JSDocInfo jSDocInfo0 = templateType0.getOwnPropertyJSDocInfo("Not declared as a constructor");
      assertNull(jSDocInfo0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createNativeAnonymousObjectType();
      ParameterizedType parameterizedType0 = jSTypeRegistry0.createParameterizedType(objectType0, objectType0);
      Visitor<AllType> visitor0 = (Visitor<AllType>) mock(Visitor.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(visitor0).caseObjectType(any(com.google.javascript.rhino.jstype.ObjectType.class));
      AllType allType0 = parameterizedType0.visit(visitor0);
      assertNull(allType0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "$/kmn");
      boolean boolean0 = templateType0.isPropertyInExterns("wXG?(1];[U9");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      JSTypeNative jSTypeNative0 = JSTypeNative.REFERENCE_ERROR_TYPE;
      JSType jSType0 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      EnumType enumType0 = new EnumType(jSTypeRegistry0, "b", (Node) null, jSType0);
      String string0 = enumType0.getDisplayName();
      assertEquals("b", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "kD`l_;xrWy'");
      ObjectType objectType0 = templateType0.getTypeOfThis();
      assertNull(objectType0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      String string0 = ObjectType.createDelegateSuffix("Unknown class name");
      assertEquals("(Unknown class name)", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, noResolvedType0);
      Node node0 = instanceObjectType0.getPropertyNode("Not declared as a constructor");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSTypeNative jSTypeNative0 = JSTypeNative.URI_ERROR_TYPE;
      InstanceObjectType instanceObjectType0 = (InstanceObjectType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      PrototypeObjectType prototypeObjectType0 = (PrototypeObjectType)instanceObjectType0.getParentScope();
      assertTrue(prototypeObjectType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      ObjectType.Property objectType_Property0 = noResolvedType0.getOwnSlot(")");
      assertNull(objectType_Property0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", (-2734), (-2734));
      ObjectType.Property objectType_Property0 = namedType0.getOwnSlot("");
      assertNull(objectType_Property0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSType jSType0 = jSTypeRegistry0.createNamedType(")", ")", 0, (-599));
      JSType[] jSTypeArray0 = new JSType[0];
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType(jSType0, jSTypeArray0);
      JSDocInfo jSDocInfo0 = functionType0.getJSDocInfo();
      assertNull(jSDocInfo0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      JSTypeNative jSTypeNative0 = JSTypeNative.REFERENCE_ERROR_TYPE;
      JSType jSType0 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      EnumType enumType0 = new EnumType(jSTypeRegistry0, "b", (Node) null, jSType0);
      boolean boolean0 = enumType0.detectImplicitPrototypeCycle();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoType noType0 = new NoType(jSTypeRegistry0);
      boolean boolean0 = noType0.hasDisplayName();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, ".equivalentTo(");
      boolean boolean0 = errorFunctionType0.hasDisplayName();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      JSTypeNative jSTypeNative0 = JSTypeNative.EVAL_ERROR_TYPE;
      InstanceObjectType instanceObjectType0 = (InstanceObjectType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      EnumType enumType0 = new EnumType(jSTypeRegistry0, "Named type with empty name component", (Node) null, instanceObjectType0);
      enumType0.testForEquality(instanceObjectType0);
      assertTrue(instanceObjectType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSTypeNative jSTypeNative0 = JSTypeNative.VOID_TYPE;
      JSType jSType0 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      EnumType enumType0 = new EnumType(jSTypeRegistry0, "Not declared as a type name", (Node) null, jSType0);
      enumType0.testForEquality(jSType0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      JSTypeNative jSTypeNative0 = JSTypeNative.REFERENCE_ERROR_TYPE;
      JSType jSType0 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      JSType[] jSTypeArray0 = new JSType[0];
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType(jSType0, false, jSTypeArray0);
      EnumType enumType0 = new EnumType(jSTypeRegistry0, "b", (Node) null, jSType0);
      boolean boolean0 = functionType0.defineInferredProperty("None", enumType0, (Node) null);
      assertTrue(functionType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.autobox();
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType(jSType0, (List<JSType>) linkedList0);
      ParameterizedType parameterizedType0 = new ParameterizedType((JSTypeRegistry) null, functionType0, jSType0);
      parameterizedType0.findPropertyType("`.\u0003iF14fUSr,d");
      assertTrue(functionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoType noType0 = new NoType(jSTypeRegistry0);
      JSType jSType0 = noType0.findPropertyType("");
      assertTrue(jSType0.canBeCalled());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      errorFunctionType0.testForEquality(errorFunctionType0);
      boolean boolean0 = errorFunctionType0.hasCachedValues();
      assertTrue(errorFunctionType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[1];
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      ObjectType.Property objectType_Property0 = new ObjectType.Property("", jSTypeArray0[0], false, node0);
      ObjectType.Property objectType_Property1 = objectType_Property0.getDeclaration();
      assertNotNull(objectType_Property1);
      assertFalse(objectType_Property1.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      JSTypeNative jSTypeNative0 = JSTypeNative.NO_RESOLVED_TYPE;
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      ObjectType.Property objectType_Property0 = new ObjectType.Property("Unknown class name", functionType0, false, (Node) null);
      ObjectType.Property objectType_Property1 = objectType_Property0.getDeclaration();
      assertFalse(objectType_Property0.isTypeInferred());
      assertNull(objectType_Property1);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "PF o[zv Aq");
      EnumElementType enumElementType0 = new EnumElementType(jSTypeRegistry0, errorFunctionType0, "Named type with empty name component");
      Node node0 = new Node(0, 1, 0);
      EnumType enumType0 = new EnumType(jSTypeRegistry0, "{dcE/s`NA)", node0, enumElementType0);
      boolean boolean0 = enumType0.defineDeclaredProperty("Named type with empty name component", errorFunctionType0, node0);
      assertTrue(boolean0);
      
      ProxyObjectType proxyObjectType0 = new ProxyObjectType(jSTypeRegistry0, enumType0);
      IndexedType indexedType0 = new IndexedType(jSTypeRegistry0, proxyObjectType0, errorFunctionType0);
      boolean boolean1 = indexedType0.isPropertyInExterns("Named type with empty name component");
      assertFalse(boolean1 == boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSTypeNative jSTypeNative0 = JSTypeNative.URI_ERROR_TYPE;
      JSType jSType0 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      EnumType enumType0 = new EnumType(jSTypeRegistry0, "Not declared as a type name", (Node) null, jSType0);
      boolean boolean0 = enumType0.defineElement("Not declared as a type name", (Node) null);
      IndexedType indexedType0 = new IndexedType(jSTypeRegistry0, enumType0, jSType0);
      boolean boolean1 = indexedType0.isPropertyInExterns("Not declared as a type name");
      assertFalse(boolean1 == boolean0);
      assertFalse(boolean1);
  }
}
/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:40:23 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.InjectableValues;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.ClassKey;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.HierarchicType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeModifier;
import com.fasterxml.jackson.databind.type.TypeParser;
import java.io.DataInputStream;
import java.lang.reflect.Array;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeFactory_ESTest extends TypeFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      Class<?> class1 = collectionType0.getParameterSource();
      Class<MapType> class2 = MapType.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class1, (Class<?>) class2);
      assertNotSame(collectionType0, javaType0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeModifier[] typeModifierArray0 = new TypeModifier[6];
      TypeFactory typeFactory0 = new TypeFactory((TypeParser) null, typeModifierArray0);
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifierArray0[5]);
      assertNotSame(typeFactory0, typeFactory1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      MapperFeature mapperFeature0 = MapperFeature.USE_STATIC_TYPING;
      objectMapper0.configure(mapperFeature0, true);
      Class<SettableBeanProperty> class0 = SettableBeanProperty.class;
      // Undeclared exception!
      objectMapper0.canSerialize(class0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      JavaType[] javaTypeArray0 = new JavaType[2];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class0, javaTypeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class0);
      assertFalse(mapType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      typeFactory0.clearCache();
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayType> class0 = ArrayType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      assertEquals(1, collectionLikeType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      Class<Integer> class1 = Integer.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      assertFalse(collectionType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<SimpleType> class0 = SimpleType.class;
      Class<Module> class1 = Module.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class1, class0);
      assertFalse(collectionLikeType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class0, javaTypeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.CollectionLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      assertFalse(arrayType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      // Undeclared exception!
      try { 
        typeFactory0.constructType((TypeReference<?>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      TypeModifier[] typeModifierArray0 = new TypeModifier[6];
      TypeFactory typeFactory0 = new TypeFactory((TypeParser) null, typeModifierArray0);
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      JavaType javaType0 = mapType0.containedTypeOrUnknown((-3799));
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(javaType0, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<ArrayList> class0 = ArrayList.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      // Undeclared exception!
      try { 
        typeFactory0.constructCollectionLikeType(class0, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.CollectionLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<JsonToken> class0 = JsonToken.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertFalse(mapLikeType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      try { 
        typeFactory0.constructFromCanonical("&P<PUH^TH");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type '&P<PUH^TH' (remaining: '<PUH^TH'): Can not locate class '&P', problem: Class '&P.class' should be in target project, but could not be found!
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, (Class<?>[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType((Class<?>) class0, (JavaType) mapType0, (JavaType) mapType0);
      assertTrue(mapLikeType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      // Undeclared exception!
      try { 
        typeFactory0.constructArrayType((JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.ArrayType", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      assertFalse(typeFactory1.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[0];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory2 = typeFactory1.withModifier(typeModifier0);
      assertFalse(typeFactory2.equals((Object)typeFactory1));
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      HierarchicType hierarchicType0 = typeFactory0._cachedHashMapType;
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<ArrayList> class0 = ArrayList.class;
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0._unknownType();
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      assertTrue(javaType1.isConcrete());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LongNode> class0 = LongNode.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      JavaType javaType0 = typeFactory0.constructSpecializedType(simpleType0, class0);
      assertSame(simpleType0, javaType0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      Class<MapType> class1 = MapType.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(javaType0, class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.type.MapType is not assignable to java.util.HashMap
         //
         verifyException("com.fasterxml.jackson.databind.JavaType", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<JsonToken> class0 = JsonToken.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.core.JsonToken is not assignable to long
         //
         verifyException("com.fasterxml.jackson.databind.JavaType", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<ArrayList> class0 = ArrayList.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      TypeFactory typeFactory0 = TypeFactory.instance;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.lang.Class not subtype of [simple type, class long]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Integer> class0 = Integer.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      JavaType javaType0 = typeFactory0.constructParametricType(class0, javaTypeArray0);
      JavaType[] javaTypeArray1 = typeFactory0.findTypeParameters(javaType0, class0);
      assertNull(javaTypeArray1);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      // Undeclared exception!
      try { 
        typeFactory0.findTypeParameters((JavaType) simpleType0, (Class<?>) class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class long is not a subtype of java.lang.Integer
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      Class<String> class0 = String.class;
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      Class<Object> class1 = Object.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(javaType0, class1);
      assertNull(javaTypeArray0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      JavaType javaType0 = typeFactory0.moreSpecificType(simpleType0, simpleType0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.moreSpecificType((JavaType) null, (JavaType) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<SimpleType> class0 = SimpleType.class;
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, class0);
      JavaType javaType0 = typeFactory0.moreSpecificType(typeBindings0.UNBOUND, (JavaType) null);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = simpleType0.forcedNarrowBy(class0);
      JavaType javaType1 = typeFactory0.moreSpecificType(javaType0, simpleType0);
      assertFalse(javaType1.isAbstract());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, mapType0);
      JavaType javaType0 = typeFactory0.moreSpecificType(typeBindings0.UNBOUND, mapType0);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      // Undeclared exception!
      try { 
        typeFactory0.constructType((Type) null, (Class<?>) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (JavaType) mapType0);
      assertFalse(javaType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (JavaType) null);
      assertTrue(javaType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Class<SettableBeanProperty> class0 = SettableBeanProperty.class;
      SimpleType simpleType0 = new SimpleType(class0);
      Class<?> class1 = TypeFactory.rawClass(simpleType0);
      assertEquals("class com.fasterxml.jackson.databind.deser.SettableBeanProperty", class1.toString());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[0];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      Class<SimpleType> class1 = SimpleType.class;
      MapLikeType mapLikeType0 = typeFactory1.constructMapLikeType(class0, class0, class1);
      assertFalse(mapLikeType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class0 = Object.class;
      Class<CollectionType>[] classArray0 = (Class<CollectionType>[]) Array.newInstance(Class.class, 1);
      Class<CollectionType> class1 = CollectionType.class;
      classArray0[0] = class1;
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class0, classArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Parameter type mismatch for java.lang.Object (and target java.lang.Object): expected 0 parameters, was given 1
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<DataInputStream> class1 = DataInputStream.class;
      Class<MapLikeType>[] classArray0 = (Class<MapLikeType>[]) Array.newInstance(Class.class, 0);
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class1, classArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Need exactly 2 parameter types for Map types (java.util.HashMap)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class0, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Need exactly 1 parameter type for Collection types (java.util.ArrayList)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      Class<?> class0 = simpleType0.getRawClass();
      JavaType javaType0 = typeFactory0.constructType((Type) class0, class0);
      assertSame(javaType0, simpleType0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<SettableBeanProperty> class0 = SettableBeanProperty.class;
      boolean boolean0 = objectMapper0.canSerialize(class0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<JsonToken> class0 = JsonToken.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      assertTrue(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<JsonToken> class0 = JsonToken.class;
      LinkedList<JavaType> linkedList0 = new LinkedList<JavaType>();
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, linkedList0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      assertEquals(2, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, class0);
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      arrayList0.add(typeBindings0.UNBOUND);
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      assertEquals(2, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, class0);
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      arrayList0.add(typeBindings0.UNBOUND);
      arrayList0.add(typeBindings0.UNBOUND);
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, stack0);
      assertTrue(javaType0.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Class<ArrayList> class0 = ArrayList.class;
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, class0);
      LinkedList<JavaType> linkedList0 = new LinkedList<JavaType>();
      linkedList0.add(typeBindings0.UNBOUND);
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, linkedList0);
      assertTrue(javaType0.hasGenericTypes());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Stack<JavaType> stack0 = new Stack<JavaType>();
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      stack0.add((JavaType) simpleType0);
      Class<MapType> class0 = MapType.class;
      // Undeclared exception!
      try { 
        typeFactory0._fromParameterizedClass(class0, stack0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Parameter type mismatch for com.fasterxml.jackson.databind.type.MapType (and target com.fasterxml.jackson.databind.type.MapType): expected 0 parameters, was given 1
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      HierarchicType hierarchicType0 = typeFactory0._cachedHashMapType;
      Class<CollectionType> class0 = CollectionType.class;
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, class0);
      JavaType javaType0 = typeFactory0._resolveVariableViaSubTypes(hierarchicType0, "K", typeBindings0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<SimpleType> class1 = SimpleType.class;
      HierarchicType hierarchicType0 = typeFactory0._findSuperTypeChain(class0, class1);
      assertNull(hierarchicType0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      Class<ClassKey> class1 = ClassKey.class;
      HierarchicType hierarchicType0 = typeFactory0._findSuperInterfaceChain(class0, class1);
      assertNull(hierarchicType0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Class<ArrayList> class0 = ArrayList.class;
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class1 = Object.class;
      HierarchicType hierarchicType0 = typeFactory0._findSuperInterfaceChain(class0, class1);
      assertFalse(hierarchicType0.isGeneric());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      InjectableValues.Std injectableValues_Std0 = new InjectableValues.Std((Map<String, Object>) null);
      ObjectReader objectReader0 = objectMapper0.reader((InjectableValues) injectableValues_Std0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      // Undeclared exception!
      try { 
        typeFactory0._arrayListSuperInterfaceChain((HierarchicType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      // Undeclared exception!
      try { 
        typeFactory0._arrayListSuperInterfaceChain((HierarchicType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }
}
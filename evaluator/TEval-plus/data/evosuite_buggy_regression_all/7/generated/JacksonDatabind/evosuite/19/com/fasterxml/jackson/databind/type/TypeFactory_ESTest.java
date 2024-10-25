/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:55:48 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.HierarchicType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeModifier;
import com.fasterxml.jackson.databind.type.TypeParser;
import java.lang.reflect.Array;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.concurrent.atomic.AtomicReference;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeFactory_ESTest extends TypeFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      Class<ReferenceType> class1 = ReferenceType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class1);
      assertTrue(mapLikeType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      assertTrue(collectionType0.hasGenericTypes());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      GenericArrayType genericArrayType0 = mock(GenericArrayType.class, new ViolatedAssumptionAnswer());
      doReturn((Type) null).when(genericArrayType0).getGenericComponentType();
      // Undeclared exception!
      try { 
        typeFactory0._constructType(genericArrayType0, (TypeBindings) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      JavaType javaType0 = objectMapper0.constructType(class0);
      objectMapper0.readerFor(javaType0);
      assertFalse(javaType0.isFinal());
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      AtomicReference<MapType> atomicReference0 = new AtomicReference<MapType>();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(atomicReference0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertFalse(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Class<Integer> class1 = Integer.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class1, class1);
      assertFalse(mapType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      typeFactory0.clearCache();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<JsonEncoding> class0 = JsonEncoding.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      assertTrue(collectionLikeType0.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<Object> class1 = Object.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters((JavaType) collectionType0, (Class<?>) class1);
      assertNull(javaTypeArray0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapLikeType> class0 = MapLikeType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      assertEquals(1, collectionLikeType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class0 = Object.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      assertFalse(arrayType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
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
  public void test13()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      MapLikeType mapLikeType0 = mapType0.withContentValueHandler(typeFactory0);
      Class<?> class1 = mapLikeType0.getParameterSource();
      Class<String> class2 = String.class;
      typeFactory0.constructType((Type) class1, (Class<?>) class2);
      Class<Integer> class3 = Integer.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class3, (JavaType[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType((Class<?>) class0, (JavaType) collectionType0);
      assertFalse(collectionLikeType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayNode> class0 = ArrayNode.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertTrue(mapLikeType0.hasGenericTypes());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      try { 
        typeFactory0.constructFromCanonical("X?w^\"`H G+vlV+");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type 'X?w^\"`H G+vlV+' (remaining: ''): Can not locate class 'X?w^\"`H G+vlV+', problem: Class 'X?w^\"`H G+vlV+.class' should be in target project, but could not be found!
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      // Undeclared exception!
      try { 
        objectMapper0.writeValueAsBytes(jsonFactory0);
        fail("Expecting exception: NoClassDefFoundError");
      
      } catch(NoClassDefFoundError e) {
         //
         // com/fasterxml/jackson/databind/JsonMappingException$Reference
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.BeanSerializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      Class<JsonEncoding>[] classArray0 = (Class<JsonEncoding>[]) Array.newInstance(Class.class, 0);
      JavaType javaType0 = typeFactory0.constructParametricType(class0, classArray0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Class<MapType> class0 = MapType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType((Class<?>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      assertFalse(mapLikeType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
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
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeFactory typeFactory1 = typeFactory0.withModifier((TypeModifier) null);
      assertNotSame(typeFactory1, typeFactory0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[0];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory2 = typeFactory1.withModifier(typeModifier0);
      assertNotSame(typeFactory0, typeFactory2);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Class<?> class0 = TypeFactory.rawClass(simpleType0);
      assertFalse(class0.isInterface());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      Class<?> class1 = TypeFactory.rawClass(class0);
      assertEquals("class java.util.HashMap", class1.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      JavaType javaType0 = objectMapper0.constructType(class0);
      javaTypeArray0[0] = javaType0;
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaTypeArray0[0], class0);
      assertFalse(javaType1.isFinal());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Class<Integer> class0 = Integer.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.lang.Integer is not assignable to int
         //
         verifyException("com.fasterxml.jackson.databind.JavaType", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.lang.Class not subtype of [simple type, class int]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      Class<Object> class1 = Object.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      Integer integer0 = new Integer((-3));
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class1, (JavaType) collectionType0, (Object) integer0, (Object) class1);
      SimpleType simpleType0 = referenceType0.withValueHandler(referenceType0);
      JavaType javaType0 = typeFactory0.constructSpecializedType(simpleType0, class0);
      assertTrue(javaType0.hasValueHandler());
      assertFalse(collectionType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      Class<Object> class1 = Object.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      Integer integer0 = new Integer((-3));
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class1, (JavaType) collectionType0, (Object) integer0, (Object) class1);
      ReferenceType referenceType1 = referenceType0.withTypeHandler(class0);
      JavaType javaType0 = typeFactory0.constructSpecializedType(referenceType1, class0);
      assertTrue(javaType0.equals((Object)collectionType0));
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Integer> class0 = Integer.class;
      Class<Object>[] classArray0 = (Class<Object>[]) Array.newInstance(Class.class, 0);
      JavaType javaType0 = typeFactory0.constructParametrizedType(class0, class0, classArray0);
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(javaType0, class0);
      assertFalse(javaType0.useStaticType());
      assertNull(javaTypeArray0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<JsonEncoding> class0 = JsonEncoding.class;
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, class0);
      // Undeclared exception!
      try { 
        typeFactory0.findTypeParameters(typeBindings0.UNBOUND, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.lang.Object is not a subtype of com.fasterxml.jackson.core.JsonEncoding
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) collectionType0, (Object) collectionType0, (Object) collectionType0);
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, referenceType0);
      JavaType javaType0 = typeFactory0.moreSpecificType(typeBindings0.UNBOUND, referenceType0);
      assertSame(referenceType0, javaType0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.moreSpecificType((JavaType) null, (JavaType) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class0 = Object.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      JavaType javaType1 = typeFactory0.moreSpecificType(javaType0, (JavaType) null);
      assertFalse(javaType1.isArrayType());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, class0);
      JavaType javaType0 = typeFactory0.moreSpecificType(typeBindings0.UNBOUND, typeBindings0.UNBOUND);
      assertFalse(javaType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class0 = Object.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      JavaType javaType1 = typeFactory0.moreSpecificType(simpleType0, javaType0);
      assertTrue(javaType1.isFinal());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Object> class0 = Object.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) null);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      JavaType javaType0 = typeFactory0.constructType((Type) collectionType0, (JavaType) collectionType0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      // Undeclared exception!
      try { 
        typeFactory0.constructType((Type) null, (JavaType) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, class0);
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[0];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      JavaType javaType0 = typeFactory1.constructSpecializedType(typeBindings0.UNBOUND, class0);
      assertFalse(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[0];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      Class<LinkedList> class0 = LinkedList.class;
      Class<ObjectIdGenerators.IntSequenceGenerator> class1 = ObjectIdGenerators.IntSequenceGenerator.class;
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, class1);
      JavaType javaType0 = typeFactory1.constructType((Type) class0, typeBindings0);
      assertFalse(javaType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JavaType) null, (JavaType) null, (JavaType) null, (JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      // Undeclared exception!
      try { 
        typeFactory1.constructCollectionType(class0, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<SimpleType> class0 = SimpleType.class;
      Class<MapLikeType> class1 = MapLikeType.class;
      Class<MapLikeType>[] classArray0 = (Class<MapLikeType>[]) Array.newInstance(Class.class, 2);
      classArray0[0] = class1;
      classArray0[1] = class1;
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class1, classArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Parameter type mismatch for com.fasterxml.jackson.databind.type.SimpleType (and target com.fasterxml.jackson.databind.type.MapLikeType): expected 0 parameters, was given 2
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Need exactly 2 parameter types for Map types (java.util.HashMap)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      JavaType[] javaTypeArray0 = new JavaType[2];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, javaTypeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, javaTypeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.CollectionLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      JavaType[] javaTypeArray0 = new JavaType[10];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Need exactly 1 parameter type for Collection types (java.util.ArrayList)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Long> class0 = Long.TYPE;
      Class<CollectionType> class1 = CollectionType.class;
      SimpleType simpleType0 = new SimpleType(class1);
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, simpleType0);
      JavaType javaType0 = typeFactory0._constructType(class0, typeBindings0);
      assertTrue(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      Class<JsonEncoding> class1 = JsonEncoding.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      assertFalse(collectionType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      Class<LinkedList> class0 = LinkedList.class;
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      Class<JsonEncoding> class0 = JsonEncoding.class;
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      arrayList0.add((JavaType) simpleType0);
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      arrayList0.add((JavaType) mapType0);
      arrayList0.add((JavaType) mapType0);
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      assertFalse(javaType0.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      Class<ArrayNode> class0 = ArrayNode.class;
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      assertFalse(javaType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      arrayList0.add((JavaType) mapLikeType0);
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, arrayList0);
      assertTrue(javaType0.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      arrayList0.add((JavaType) mapLikeType0);
      Class<ArrayNode> class1 = ArrayNode.class;
      // Undeclared exception!
      try { 
        typeFactory0._fromParameterizedClass(class1, arrayList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Parameter type mismatch for com.fasterxml.jackson.databind.node.ArrayNode (and target com.fasterxml.jackson.databind.node.ArrayNode): expected 0 parameters, was given 1
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      Class<Object> class1 = Object.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      Class<?> class2 = collectionType0.getParameterSource();
      Class<ReferenceType> class3 = ReferenceType.class;
      typeFactory0.constructType((Type) class2, (Class<?>) class3);
      assertFalse(collectionType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      HierarchicType hierarchicType0 = new HierarchicType(class0);
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, collectionType0);
      JavaType javaType0 = typeFactory0._resolveVariableViaSubTypes(hierarchicType0, "", typeBindings0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<SimpleType> class0 = SimpleType.class;
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, class0);
      JavaType javaType0 = typeFactory0._resolveVariableViaSubTypes((HierarchicType) null, "K", typeBindings0);
      assertFalse(javaType0.hasGenericTypes());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Integer> class0 = Integer.class;
      Class<MapLikeType> class1 = MapLikeType.class;
      HierarchicType hierarchicType0 = typeFactory0._findSuperTypeChain(class0, class1);
      assertNull(hierarchicType0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      Class<Integer> class1 = Integer.class;
      HierarchicType hierarchicType0 = typeFactory0._findSuperInterfaceChain(class0, class1);
      assertNull(hierarchicType0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      Class<Object> class1 = Object.class;
      HierarchicType hierarchicType0 = typeFactory0._findSuperInterfaceChain(class0, class1);
      assertFalse(hierarchicType0.isGeneric());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      HierarchicType hierarchicType0 = new HierarchicType(class0);
      typeFactory0._arrayListSuperInterfaceChain(hierarchicType0);
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

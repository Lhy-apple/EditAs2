/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:33:50 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.module.SimpleModule;
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
import java.lang.reflect.Array;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.time.chrono.ThaiBuddhistEra;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Stack;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeFactory_ESTest extends TypeFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, mapType0);
      typeFactory0.constructType((Type) class0, typeBindings0);
      HierarchicType hierarchicType0 = typeFactory0._cachedHashMapType;
      ParameterizedType parameterizedType0 = hierarchicType0.asGeneric();
      JavaType javaType0 = typeFactory0.constructType((Type) parameterizedType0);
      assertTrue(javaType0.hasGenericTypes());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      JavaType javaType0 = typeFactory0.moreSpecificType(collectionType0, simpleType0);
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapType> class0 = MapType.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      JavaType javaType0 = typeFactory0.constructParametricType(class0, javaTypeArray0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonAutoDetect.Visibility jsonAutoDetect_Visibility0 = JsonAutoDetect.Visibility.NON_PRIVATE;
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      PropertyAccessor propertyAccessor0 = PropertyAccessor.FIELD;
      ObjectMapper objectMapper1 = objectMapper0.setVisibility(propertyAccessor0, jsonAutoDetect_Visibility0);
      Class<CreatorProperty> class0 = CreatorProperty.class;
      ObjectReader objectReader0 = objectMapper1.reader((Class<?>) class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ThaiBuddhistEra> class0 = ThaiBuddhistEra.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class0);
      assertTrue(mapType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapType> class0 = MapType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      assertTrue(mapLikeType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      typeFactory0.clearCache();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ObjectReader> class0 = ObjectReader.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      assertFalse(collectionLikeType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JavaType javaType0 = typeFactory0.moreSpecificType(simpleType0, collectionType0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<SimpleModule> class0 = SimpleModule.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      assertFalse(collectionLikeType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<SimpleType> class0 = SimpleType.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      assertFalse(arrayType0.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
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
  public void test13()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<SimpleModule> class0 = SimpleModule.class;
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
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertTrue(mapLikeType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      try { 
        typeFactory0.constructFromCanonical(")");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type ')' (remaining: ''): Can not locate class ')', problem: Class ').class' should be in target project, but could not be found!
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<SettableBeanProperty> class0 = SettableBeanProperty.class;
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
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType((Class<?>) class0, (JavaType) mapType0, (JavaType) mapType0);
      assertTrue(mapLikeType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
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
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      assertFalse(typeFactory1.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeFactory typeFactory1 = typeFactory0.withModifier((TypeModifier) null);
      assertFalse(typeFactory1.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[0];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory2 = typeFactory1.withModifier(typeModifier0);
      assertFalse(typeFactory2.equals((Object)typeFactory1));
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      Class<?> class0 = TypeFactory.rawClass(simpleType0);
      assertEquals(17, class0.getModifiers());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<ClassKey> class0 = ClassKey.class;
      Class<?> class1 = TypeFactory.rawClass(class0);
      assertFalse(class1.isAnnotation());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      JavaType javaType0 = typeFactory0.constructSpecializedType(mapType0, class0);
      assertSame(javaType0, mapType0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      JavaType javaType0 = typeFactory0._unknownType();
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      assertFalse(javaType1.isAbstract());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      Class<SettableBeanProperty> class0 = SettableBeanProperty.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.deser.SettableBeanProperty is not assignable to long
         //
         verifyException("com.fasterxml.jackson.databind.JavaType", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
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
  public void test28()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      Class<ThaiBuddhistEra> class1 = ThaiBuddhistEra.class;
      Class<CollectionLikeType>[] classArray0 = (Class<CollectionLikeType>[]) Array.newInstance(Class.class, 0);
      JavaType javaType0 = typeFactory0.constructParametrizedType(class0, class1, classArray0);
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(javaType0, class1);
      assertFalse(javaType0.useStaticType());
      assertNull(javaTypeArray0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      Class<ThaiBuddhistEra> class1 = ThaiBuddhistEra.class;
      // Undeclared exception!
      try { 
        typeFactory0.findTypeParameters((JavaType) collectionType0, (Class<?>) class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.util.LinkedList is not a subtype of java.time.chrono.ThaiBuddhistEra
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ThaiBuddhistEra> class0 = ThaiBuddhistEra.class;
      Class<Object> class1 = Object.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class1);
      assertNull(javaTypeArray0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0._unknownType();
      JavaType javaType1 = typeFactory0.moreSpecificType(javaType0, (JavaType) null);
      assertFalse(javaType1.isFinal());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.moreSpecificType((JavaType) null, (JavaType) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      JavaType javaType0 = typeFactory0.moreSpecificType(simpleType0, simpleType0);
      assertFalse(javaType0.hasGenericTypes());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, mapType0);
      JavaType javaType0 = typeFactory0.constructType((Type) typeBindings0.UNBOUND, (Class<?>) class0);
      assertFalse(javaType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) null);
      assertTrue(javaType0.hasGenericTypes());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Long> class0 = Long.TYPE;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (JavaType) simpleType0);
      assertTrue(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      MapType mapType0 = MapType.construct(class0, simpleType0, simpleType0);
      Class<?> class1 = mapType0.getParameterSource();
      JavaType javaType0 = typeFactory0.constructType((Type) class1, (JavaType) null);
      assertFalse(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      // Undeclared exception!
      try { 
        typeFactory0.constructType((Type) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      LinkedList<JavaType> linkedList0 = new LinkedList<JavaType>();
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[2];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      // Undeclared exception!
      try { 
        typeFactory1._fromParameterizedClass(class0, linkedList0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[5];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory1.constructRawCollectionType(class0);
      Class<?> class1 = collectionType0.getParameterSource();
      Class<String> class2 = String.class;
      JavaType javaType0 = typeFactory1.constructType((Type) class1, (Class<?>) class2);
      assertFalse(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ClassKey> class0 = ClassKey.class;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[0];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      JavaType javaType0 = typeFactory1.constructType((Type) class0);
      assertFalse(javaType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayType>[] classArray0 = (Class<ArrayType>[]) Array.newInstance(Class.class, 0);
      Class<ThaiBuddhistEra> class0 = ThaiBuddhistEra.class;
      Class<JsonSerializer> class1 = JsonSerializer.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class1, classArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Parameter type mismatch for java.time.chrono.ThaiBuddhistEra (and target com.fasterxml.jackson.databind.JsonSerializer): expected 1 parameters, was given 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      Class<CreatorProperty>[] classArray0 = (Class<CreatorProperty>[]) Array.newInstance(Class.class, 2);
      Class<CreatorProperty> class1 = CreatorProperty.class;
      classArray0[0] = class1;
      classArray0[1] = classArray0[0];
      JavaType javaType0 = typeFactory0.constructParametrizedType(class0, class0, classArray0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Class<CollectionType> class1 = CollectionType.class;
      Class<CreatorProperty>[] classArray0 = (Class<CreatorProperty>[]) Array.newInstance(Class.class, 0);
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
  public void test45()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      Class<SimpleModule>[] classArray0 = (Class<SimpleModule>[]) Array.newInstance(Class.class, 1);
      Class<SimpleModule> class1 = SimpleModule.class;
      classArray0[0] = class1;
      JavaType javaType0 = typeFactory0.constructParametrizedType(class0, class0, classArray0);
      assertFalse(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      Class<LinkedList> class1 = LinkedList.class;
      Class<ArrayType>[] classArray0 = (Class<ArrayType>[]) Array.newInstance(Class.class, 0);
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class1, class0, classArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Need exactly 1 parameter type for Collection types (java.util.LinkedList)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      ObjectMapper objectMapper1 = objectMapper0.setTypeFactory(typeFactory0);
      Class<CreatorProperty> class0 = CreatorProperty.class;
      ObjectReader objectReader0 = objectMapper1.reader((Class<?>) class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ThaiBuddhistEra> class0 = ThaiBuddhistEra.class;
      Vector<JavaType> vector0 = new Vector<JavaType>();
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, vector0);
      assertFalse(javaType0.hasGenericTypes());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, stack0);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      Stack<JavaType> stack0 = new Stack<JavaType>();
      stack0.add((JavaType) mapType0);
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, stack0);
      assertFalse(javaType0.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      Stack<JavaType> stack0 = new Stack<JavaType>();
      stack0.add((JavaType) mapType0);
      stack0.add((JavaType) mapType0);
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, stack0);
      assertFalse(javaType0.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Vector<JavaType> vector0 = new Vector<JavaType>();
      Class<SimpleType> class0 = SimpleType.class;
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, vector0);
      assertEquals(0, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      Stack<JavaType> stack0 = new Stack<JavaType>();
      stack0.add((JavaType) collectionType0);
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, stack0);
      assertFalse(javaType0.equals((Object)collectionType0));
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Vector<JavaType> vector0 = new Vector<JavaType>();
      vector0.add((JavaType) null);
      Class<SimpleType> class0 = SimpleType.class;
      // Undeclared exception!
      try { 
        typeFactory0._fromParameterizedClass(class0, vector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Parameter type mismatch for com.fasterxml.jackson.databind.type.SimpleType (and target com.fasterxml.jackson.databind.type.SimpleType): expected 0 parameters, was given 1
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, collectionType0);
      JavaType javaType0 = typeFactory0._resolveVariableViaSubTypes((HierarchicType) null, "g/B2#5@'$|xm2vIK7z>a", typeBindings0);
      assertFalse(javaType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, mapType0);
      JavaType javaType0 = typeFactory0._resolveVariableViaSubTypes((HierarchicType) null, "V", typeBindings0);
      assertFalse(javaType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      HierarchicType hierarchicType0 = new HierarchicType(class0);
      JavaType javaType0 = typeFactory0._resolveVariableViaSubTypes(hierarchicType0, "", (TypeBindings) null);
      assertFalse(javaType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0._resolveVariableViaSubTypes((HierarchicType) null, " since it is not abstract", (TypeBindings) null);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      Class<Object> class1 = Object.class;
      HierarchicType hierarchicType0 = typeFactory0._findSuperInterfaceChain(class0, class1);
      assertFalse(hierarchicType0.isGeneric());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapType> class0 = MapType.class;
      Class<ThaiBuddhistEra> class1 = ThaiBuddhistEra.class;
      HierarchicType hierarchicType0 = typeFactory0._findSuperInterfaceChain(class0, class1);
      assertNull(hierarchicType0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, stack0);
      assertFalse(javaType0.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
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

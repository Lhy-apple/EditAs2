/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:03:44 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.introspect.TypeResolutionContext;
import com.fasterxml.jackson.databind.introspect.VirtualAnnotatedMember;
import com.fasterxml.jackson.databind.node.DecimalNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.ClassStack;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeModifier;
import com.fasterxml.jackson.databind.type.TypeParser;
import java.lang.reflect.Array;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Properties;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeFactory_ESTest extends TypeFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapType> class0 = MapType.class;
      Class<CollectionLikeType> class1 = CollectionLikeType.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      JavaType javaType0 = typeFactory0.constructSimpleType(class0, class1, javaTypeArray0);
      assertTrue(javaType0.isFinal());
      assertNotNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType((Class<?>) class0, (JavaType) collectionType0);
      assertFalse(collectionLikeType0.equals((Object)collectionType0));
      assertNotSame(collectionLikeType0, collectionType0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, (TypeModifier[]) null);
      assertFalse(typeFactory1.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Object> class0 = Object.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      Class<RuntimeException>[] classArray0 = (Class<RuntimeException>[]) Array.newInstance(Class.class, 0);
      JavaType javaType0 = typeFactory0.constructParametricType(class0, classArray0);
      assertFalse(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapLikeType((Class<?>) null, (Class<?>) null, (Class<?>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType[] javaTypeArray0 = new JavaType[4];
      Class<Integer> class0 = Integer.class;
      Class<CollectionType> class1 = CollectionType.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class1, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.lang.Integer with 4 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapperFeature> class0 = MapperFeature.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class0, (Class<?>[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<CreatorProperty> class0 = CreatorProperty.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      typeFactory0.clearCache();
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Integer> class0 = Integer.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      assertTrue(collectionLikeType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      assertFalse(collectionLikeType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      assertFalse(arrayType0.isAbstract());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
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
  public void test14()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructRawMapType(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      JavaType javaType0 = typeParser0.parse("com.fasterxml.jackson.annotation.JsonAutoDetect$Visibility");
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(javaType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping not subtype of [simple type, class com.fasterxml.jackson.annotation.JsonAutoDetect$Visibility]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      try { 
        typeFactory0.constructFromCanonical("PvuRG92TlqH.,E0\"b&b");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type 'PvuRG92TlqH.,E0\"b&b' (remaining: ',E0\"b&b'): Can not locate class 'PvuRG92TlqH.', problem: Class 'PvuRG92TlqH/.class' should be in target project, but could not be found!
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      ClassLoader classLoader0 = ClassLoader.getSystemClassLoader();
      TypeFactory typeFactory1 = typeFactory0.withClassLoader(classLoader0);
      try { 
        typeFactory1.findClass("pNe'0'v>HT5`6)dbduA");
        fail("Expecting exception: ClassNotFoundException");
      
      } catch(ClassNotFoundException e) {
         //
         // pNe'0'v>HT5`6)dbduA
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<DeserializationFeature> class0 = DeserializationFeature.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class0, (TypeBindings) null);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ReferenceType> class0 = ReferenceType.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JavaType javaType0 = typeFactory0.constructReferenceType(class0, simpleType0);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapLikeType> class0 = MapLikeType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType((Class<?>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      typeFactory0.constructArrayType((JavaType) mapLikeType0);
      assertFalse(mapLikeType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      TypeModifier typeModifier1 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory2 = typeFactory1.withModifier(typeModifier1);
      assertNotSame(typeFactory0, typeFactory2);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeFactory typeFactory1 = typeFactory0.withModifier((TypeModifier) null);
      assertNotSame(typeFactory1, typeFactory0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertNotNull(javaType0);
      
      Class<?> class1 = TypeFactory.rawClass(javaType0);
      assertEquals("class com.fasterxml.jackson.databind.type.ResolvedRecursiveType", class1.toString());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Boolean> class0 = Boolean.TYPE;
      Class<?> class1 = TypeFactory.rawClass(class0);
      assertFalse(class1.isInterface());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      try { 
        typeFactory0.findClass("nT5+$Tq]KIEVu3");
        fail("Expecting exception: ClassNotFoundException");
      
      } catch(ClassNotFoundException e) {
         //
         // Class 'nT5+$Tq]KIEVu3.class' should be in target project, but could not be found!
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("void");
      assertEquals("void", class0.toString());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0._findPrimitive("int");
      assertFalse(class0.isInterface());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0._findPrimitive("long");
      assertNotNull(class0);
      
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (JavaType) simpleType0);
      assertEquals("long", class0.toString());
      assertTrue(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("float");
      assertEquals("float", class0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("double");
      assertEquals("double", class0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("boolean");
      assertEquals("boolean", class0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("byte");
      assertEquals("byte", class0.toString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("char");
      assertEquals("char", class0.toString());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0._findPrimitive("short");
      assertNotNull(class0);
      
      typeFactory0.constructType((Type) class0, class0);
      assertEquals("short", class0.toString());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertTrue(javaType0.isEnumType());
      
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      assertSame(javaType1, javaType0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      JavaType javaType0 = typeFactory0._unknownType();
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      assertTrue(javaType1.isConcrete());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      JavaType javaType0 = typeFactory0.constructSpecializedType(simpleType0, class0);
      assertFalse(javaType0.isAbstract());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class0 = Object.class;
      Class<MapType> class1 = MapType.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class1, (JavaType) null);
      TypeResolutionContext.Basic typeResolutionContext_Basic0 = new TypeResolutionContext.Basic(typeFactory0, typeBindings0);
      Class<Properties> class2 = Properties.class;
      VirtualAnnotatedMember virtualAnnotatedMember0 = new VirtualAnnotatedMember(typeResolutionContext_Basic0, class1, ";s2-]", class2);
      JavaType javaType0 = virtualAnnotatedMember0.getType();
      JavaType javaType1 = typeFactory0.constructGeneralizedType(javaType0, class0);
      assertFalse(javaType1.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<CollectionType> class0 = CollectionType.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      JavaType javaType1 = typeFactory0.constructGeneralizedType(javaType0, class0);
      assertFalse(javaType1.isContainerType());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Internal error: class java.lang.Object not included as super-type for [simple type, class java.lang.Enum]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Properties> class0 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(javaType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.util.Properties not a super-type of [simple type, class java.lang.Object]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Class<Module> class0 = Module.class;
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<SerializationFeature> class1 = SerializationFeature.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class1, class0);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      Class<HashSet> class0 = HashSet.class;
      Class<AnnotationIntrospector.ReferenceProperty.Type> class1 = AnnotationIntrospector.ReferenceProperty.Type.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      JavaType javaType0 = typeFactory0.moreSpecificType(simpleType0, collectionType0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.moreSpecificType((JavaType) null, (JavaType) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      Class<MapperFeature> class1 = MapperFeature.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      JavaType javaType0 = typeFactory0.moreSpecificType(collectionType0, (JavaType) null);
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = TypeFactory.unknownType();
      JavaType javaType1 = typeFactory0.moreSpecificType(javaType0, javaType0);
      assertTrue(javaType1.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class0 = Object.class;
      JavaType javaType0 = typeFactory0._findWellKnownSimple(class0);
      assertNotNull(javaType0);
      
      Class<LinkedList> class1 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class1, class0);
      typeFactory0.moreSpecificType(javaType0, collectionType0);
      assertTrue(javaType0.isJavaLangObject());
      assertFalse(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) null);
      assertTrue(javaType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class0 = Object.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (JavaType) null);
      assertTrue(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapType(class0, class0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertFalse(mapLikeType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayType> class0 = ArrayType.class;
      Class<SimpleObjectIdResolver>[] classArray0 = (Class<SimpleObjectIdResolver>[]) Array.newInstance(Class.class, 1);
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, classArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<MapperFeature> class1 = MapperFeature.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) collectionType0);
      JavaType[] javaTypeArray0 = new JavaType[6];
      JavaType javaType0 = typeFactory0._constructSimple(class1, typeBindings0, collectionType0, javaTypeArray0);
      assertEquals(1, collectionType0.containedTypeCount());
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class0 = Object.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertNotNull(javaType0);
      assertTrue(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashSet> class0 = HashSet.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructType((Type) null, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = TypeFactory.unknownType();
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn(javaType0, (JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      doReturn((String) null).when(typeModifier0).toString();
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      Class<LinkedList> class0 = LinkedList.class;
      // Undeclared exception!
      try { 
        typeFactory1.constructCollectionType(class0, class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // TypeModifier null (of type com.fasterxml.jackson.databind.type.TypeModifier$MockitoMock$1817027451) return null for type [simple type, class java.lang.Object]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      doReturn((String) null).when(typeModifier0).toString();
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      Class<ArrayList> class0 = ArrayList.class;
      // Undeclared exception!
      try { 
        typeFactory1.constructRawCollectionType(class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // TypeModifier null (of type com.fasterxml.jackson.databind.type.TypeModifier$MockitoMock$1817027451) return null for type [simple type, class java.lang.Object]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<EnumSet> class0 = EnumSet.class;
      Class<DecimalNode> class1 = DecimalNode.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      assertFalse(collectionType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<RuntimeException> class0 = RuntimeException.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ParameterizedType parameterizedType0 = mock(ParameterizedType.class, new ViolatedAssumptionAnswer());
      doReturn((Type[]) null).when(parameterizedType0).getActualTypeArguments();
      doReturn((Type) null).when(parameterizedType0).getRawType();
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      // Undeclared exception!
      try { 
        typeFactory0._fromParamType((ClassStack) null, parameterizedType0, typeBindings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }
}
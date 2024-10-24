/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:43:10 GMT 2023
 */

package org.apache.commons.codec.language;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.codec.language.DoubleMetaphone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DoubleMetaphone_ESTest extends DoubleMetaphone_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("CAESAR", true);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("SSR", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      doubleMetaphone0.setMaxCodeLen(105);
      String string0 = doubleMetaphone0.encode("org.apache.commons.codec.EncoderException");
      assertNotNull(string0);
      assertEquals("ARKPXKMNSKTKNKTRKSPXN", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("+*ygH;i`mGN|jqu");
      assertNotNull(string0);
      assertEquals("KMNJ", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      doubleMetaphone0.isDoubleMetaphoneEqual("GNGNKNGH^0A5", "GNGNKNGH^0A5");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("BACHER");
      assertEquals("PKR", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("Gy}pOu>,fJN8");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("KPFN", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("g3xhYh2");
      assertEquals("KKS", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("gHwlgg");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("KLK", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("!qQrPHg{0g m'", false);
      assertEquals("KRFK", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("(LLj^vS&");
      assertEquals("LFS", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("Yz,f@gN");
      assertEquals("ASFN", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("MF1)2BbscbZ;C");
      assertEquals("MFPS", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("Ff&`@=] pO*Ye?I");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("FP", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("[KKcD");
      assertEquals("KKT", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("$NnABlp@@R9uvt.f,X");
      assertEquals("NPLP", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("3ST>Q ,w=+Sci");
      assertEquals("STKS", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("4.!w.m8Emv`vv>");
      assertEquals("MMFF", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      Object object0 = doubleMetaphone0.encode((Object) "GI");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("J", object0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      try { 
        doubleMetaphone0.encode((Object) doubleMetaphone0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // DoubleMetaphone encode parameter is not of type String
         //
         verifyException("org.apache.commons.codec.language.DoubleMetaphone", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("CHORE");
      assertEquals("XR", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("WICZ");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("AKS", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("CZ");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("S", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("CC");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("K", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      Object object0 = doubleMetaphone0.encode((Object) "UCCES");
      assertEquals("AKSS", object0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("cK}7:Jm>]Zn&`z2cZj8");
      assertEquals("KMSN", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("CE");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("S", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("CIO");
      assertEquals("S", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      Object object0 = doubleMetaphone0.encode((Object) "n_,UCCES");
      assertEquals("NKSS", object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("CCES");
      assertEquals("XS", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("CHOR`E");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("KR", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("ARCHIT");
      assertEquals("ARKT", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      doubleMetaphone0.isDoubleMetaphoneEqual("^{2FdG]`)P", "^{2FdG]`)P");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("u}O<9ddUi{?i|x");
      assertEquals("ATKS", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("$GN~:Q`RSX<H");
      assertEquals("NKRS", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("RANGER");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("RNJR", string0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("RGY");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("RJ", string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("Migy");
      assertEquals("MJ", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode(")ZGY$>");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("SK", string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("AGGI");
      assertEquals("AJ", string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("KNGH^0A5");
      assertEquals("NK", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("MighyT");
      assertEquals("MT", string0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("PNMIGH90YT");
      assertEquals("NMT", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("h7s![m&");
      assertEquals("SM", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("HEIM");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("HM", string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("GNJOSE");
      assertEquals("NJS", string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("JOSE");
      assertEquals("HS", string0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("J8");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("J", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("[yJAS%i%P{mrP/Cmf9");
      assertEquals("JSPM", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("f'oj?w95m;");
      assertEquals("FJM", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("PNYA%UAJJ-,");
      assertEquals("NJ", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("z_J");
      assertEquals("SJ", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("PSILLO");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("SL", string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("PpLLO");
      assertEquals("PL", string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("WR");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("R", string0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      Object object0 = doubleMetaphone0.encode((Object) "IER");
      assertEquals("A", object0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("fIrR");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("FR", string0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("YSL");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("AL", string0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("SUGAR");
      assertEquals("XKR", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("SSR", false);
      assertEquals("SR", string0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("SH");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("X", string0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("SIA");
      assertEquals("S", string0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("WRSIAN", true);
      assertEquals("RSN", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("SmhV&mj8");
      assertEquals("SMFM", string0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("`*NWSz.R>");
      assertEquals("NSR", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("PSCHORE", false);
      assertEquals("XR", string0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("SCH");
      assertEquals("X", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("TCH");
      assertEquals("X", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("TH");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("0", string0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("TTH");
      assertEquals("0", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("TTGH");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("TK", string0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("EwR");
      assertEquals("AR", string0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("w>`&j4<BMz,r}i-a:+");
      assertEquals("JPMS", string0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("WH");
      assertEquals("A", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("^AW");
      assertEquals("", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("OWSKI");
      assertEquals("ASK", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("GNWITZ");
      assertEquals("NTS", string0);
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("XR", false);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("SR", string0);
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("Zh>(M*lF");
      assertEquals("JMLF", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("sR5zza;LT");
      assertEquals("SRSL", string0);
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("zb[h v:Wc0]%0");
      assertEquals("SPFK", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("WITZ");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("ATS", string0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("GNCHIA");
      assertEquals("NK", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("BACH");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("PK", string0);
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("CHAE");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("X", string0);
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("[ChEI");
      assertEquals("X", string0);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test86()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("OCH");
      assertEquals("AK", string0);
  }

  @Test(timeout = 4000)
  public void test87()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.encode("UMB");
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
      assertEquals("AM", string0);
  }

  @Test(timeout = 4000)
  public void test88()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      doubleMetaphone0.encode((String) null);
      assertEquals(4, doubleMetaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test89()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      String string0 = doubleMetaphone0.doubleMetaphone("W");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test90()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      DoubleMetaphone.DoubleMetaphoneResult doubleMetaphone_DoubleMetaphoneResult0 = doubleMetaphone0.new DoubleMetaphoneResult(3);
      doubleMetaphone_DoubleMetaphoneResult0.appendPrimary('Q');
      doubleMetaphone_DoubleMetaphoneResult0.appendPrimary("KS");
      doubleMetaphone_DoubleMetaphoneResult0.append('A', '0');
      assertEquals("0", doubleMetaphone_DoubleMetaphoneResult0.getAlternate());
  }

  @Test(timeout = 4000)
  public void test91()  throws Throwable  {
      DoubleMetaphone doubleMetaphone0 = new DoubleMetaphone();
      DoubleMetaphone.DoubleMetaphoneResult doubleMetaphone_DoubleMetaphoneResult0 = doubleMetaphone0.new DoubleMetaphoneResult(3);
      doubleMetaphone_DoubleMetaphoneResult0.appendPrimary('Q');
      doubleMetaphone_DoubleMetaphoneResult0.appendPrimary("KS");
      boolean boolean0 = doubleMetaphone_DoubleMetaphoneResult0.isComplete();
      assertEquals("QKS", doubleMetaphone_DoubleMetaphoneResult0.getPrimary());
      assertFalse(boolean0);
  }
}
